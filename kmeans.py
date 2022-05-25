import polyscope as ps
import numpy as np
from wavefront import *
import random
import heapq
import time
import itertools

import sys
ID_MESH_LAST = 0

class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces
    
    def getMeshAreaCentroid(self):
        meshVolume = 0
        temp = [0,0,0]

        for face in self.faces:
            v1 = self.vertices[face[0]]
            v2 = self.vertices[face[1]]
            v3 = self.vertices[face[2]]
            vect1 = v2-v1
            vect2 = v3-v1
            center = (v1 + v2 + v3) / 3
            area = np.linalg.norm(
                np.array(
                    [
                        vect1[1]*vect2[2]-vect1[2]*vect2[1],
                        vect1[2]*vect2[0]-vect1[0]*vect2[2],
                        vect1[0]*vect2[1]-vect1[1]*vect2[0]
                        ]
                    )
                ) * .5
            meshVolume += area
            temp = center * area

        return temp / meshVolume

class Proxy:
    def __init__(self, regionIndex, faceIndexes, proxyMesh=None, proxyCenter=None, proxyNormal=None, proxyVector=None, combinaisons=None, polyMesh=None):
        if combinaisons is None:
            combinaisons = []
        self.regionIndex = regionIndex
        self.faceIndexes = faceIndexes
        self.proxyMesh = proxyMesh
        self.proxyCenter = proxyCenter
        self.proxyNormal = proxyNormal
        self.proxyVector = proxyVector
        self.combinaisons = combinaisons
        self.polyMesh = polyMesh

class QueueElement:
    def __init__(self, error, regionIndex, index):
        self.error = error
        self.regionIndex = regionIndex
        self.index = index

def KMeans(n, proxys, faceNormals, vertices, faceVertexIndexes, areaFaces, faceEdges, adjacentToFaces):
    colors = []
    for i in range(len(proxys)):
        color = Randomcolor()
        colors.append(color)
    clustersMap = []
    for i in range(0, n):
        proxys = GetProxy(proxys)
        regions = GetProxySeed(proxys, faceNormals, areaFaces) if i > 0 else proxys
        queue, assignedIndexes = BuildQueue(regions,faceNormals,areaFaces,adjacentToFaces)
        regions, worst = AssignToRegion(faceNormals,areaFaces,adjacentToFaces,regions,queue,assignedIndexes)
        splitRegions = SplitRegion(faceNormals,areaFaces,adjacentToFaces,regions,worst)
        InsertRegions(regions,splitRegions)
        adjacentRegions = FindAdjacentRegions(faceEdges, regions)
        regionsToCombine = FindRegionsToCombine(regions,adjacentRegions,faceNormals,areaFaces)
        regions = InsertRegions(regions, [regionsToCombine])
        newProxys = []
        clustersMap = []
        for region in regions:
            ps.remove_surface_mesh(region.polyMesh)
            newProxyIndexes = region.faceIndexes
            newPolyMeshName, vertexMap, newProxyMesh = GrowSeeds(region.faceIndexes,
                                                faceVertexIndexes,
                                                vertices,
                                                colors[region.regionIndex])
            newProxys.append(Proxy(region.regionIndex,
                              newProxyIndexes,
                              newProxyMesh,
                              polyMesh=newPolyMeshName))
            clustersMap.append(vertexMap)
        proxys = newProxys
    return proxys, clustersMap

def FindRegion(regions, index):
    return [region for region in regions if region.regionIndex == index][0]

def RemoveRegion(regions, index):
    return [region for region in regions if region.regionIndex != index]

def InsertRegions(regions, insertRegions):
    remIndexes = set()
    for region in insertRegions:
        remIndexes.update(region.regionIndex)

    for index in remIndexes:
        relatedRegion = FindRegion(regions,index)
        if relatedRegion.polyMesh:
            ps.remove_surface_mesh(relatedRegion.polyMesh)

    for index in remIndexes:
        regions = RemoveRegion(regions,index)

    regions.extend(insertRegions)
    return regions

def GetProxy(proxys):
    for proxy in proxys:
        try:
            proxy.proxyCenter = mesh.GetMeshAreaCentroid()
        except IndexError:
            print("Done weird things")
        proxy.proxyNormal = GetProxyNormal(proxy.faceIndexes)
    return proxys

def GetProxyNormal(indexes):
    proxyNormal = np.array([0,0,0])
    for index in indexes:
        proxyNormal = np.add(proxyNormal, 1)
    proxyNormal /= np.linalg.norm(proxyNormal)
    return proxyNormal

def GetProxySeed(proxys, faceNormals, areaFaces):
    regions = []
    for proxy in proxys:
        regionIndex = proxy.regionIndex
        faceIndexes = proxy.faceIndexes
        proxyNormal = proxy.proxyNormal

        errors = MetricError(regionIndex,
                             faceIndexes,
                             faceNormals,
                             areaFaces,
                             proxyNormal)
        errors.sort(key=lambda x: x.error)
        seedFaceIndex = errors.pop().index
        region = Proxy(proxy.regionIndex,
                  [seedFaceIndex],
                  proxy.proxyMesh,
                  proxy.proxyCenter,
                  proxy.proxyNormal)
        regions.append(region)
    return regions

def MetricError(regionIndex, faceIndexes, faceNormals, areaFaces, proxyNormal):
    errors = []
    for index in faceIndexes:
            area = areaFaces[index]
            normal = faceNormals[index]
            try:
                normalError = normal - proxyNormal
            except TypeError:
                proxyNormal = np.array([0,0,0])
                normalError = normal - proxyNormal
            moduleNormalError = (np.linalg.norm(normalError)) ** 2
            error = moduleNormalError * area
            errors.append(QueueElement(error, regionIndex, index))
    return errors

def UpdateQueue(region, faceNormals, areaFaces, queue, newFaces):
    regionIndex = region.regionIndex
    proxyNormal = region.proxyNormal
    newFacesErrors = MetricError(regionIndex,
                                 newFaces,
                                 faceNormals,
                                 areaFaces,
                                 proxyNormal)
    queue.extend(newFacesErrors)
    queue.sort(key=lambda x: x.error)
    return queue

def UpdateQueueNew(region, faceNormals, areaFaces, queue, newFaces):
    regionIndex = region.regionIndex
    proxyNormal = region.proxyNormal
    for index in newFaces:
        area = areaFaces[index]
        normal = faceNormals[index]
        try:
            normalError = normal - proxyNormal
        except TypeError:
            proxyNormal = np.array([0,0,0])
            normalError = normal - proxyNormal
        moduleNormalError = (np.linalg.norm(normalError)) ** 2
        error = moduleNormalError * area
        heapq.heappush(queue, QueueElement(error, regionIndex, index))
    return queue

def AssignToRegion(faceNormals, areaFaces, adjacentFaces, regions, queue, assignedIndexes):
    ## Container list for the items popped from the priority list.
    heapq.heapify(queue)
    globalQueue = []
    assignedIndexes = set(assignedIndexes)
    ## Until the priority queue is not empty, keep popping
    ## the item with least priority from the priority queue.
    while queue:
            mostPriority = heapq.heappop(queue)
            faceIndex =  mostPriority.index
            ## If the index of the popped face has already
            ## been assigned skip to the next one.
            if faceIndex not in assignedIndexes:
                globalQueue.append(mostPriority)
                regionIndex = mostPriority.regionIndex
                regions[regionIndex].faceIndexes.append(faceIndex)
                assignedIndexes.add(faceIndex)
                ## Get the adjacent faces of the popped face
                ## and append them to the priority queue.
                newAdjacentFaces = set(adjacentFaces[faceIndex])
                ## If an adjacent face has already been assigned
                ## to a region, skip it.
                newAdjacentFaces -= assignedIndexes
                ## Append faces to priority queue.
                queue = UpdateQueueNew(FindRegion(regions, regionIndex).regionIndex,
                                       faceNormals,
                                       areaFaces,
                                       queue,
                                       newAdjacentFaces)

    globalQueue.sort(key=lambda x: -x.error)
    worst = globalQueue.pop()

    return regions, worst

def AssignToWorstRegion(faceNormals, areaFaces, adjacentFaces, regions, queue, assignedIndexes, oldRegionFaces):
    regionDomain = frozenset(oldRegionFaces)
    assignedIndexes = set(assignedIndexes)
    queue = [i for i in queue if i.index in regionDomain]
    while queue:
        mostPriority = queue.pop()
        faceIndex = mostPriority.index
        if faceIndex not in assignedIndexes:
            regionIndex = mostPriority.regionIndex
            for region in regions:
                if regionIndex == region.regionIndex:
                    region.faceIndexes.append(faceIndex)
                    assignedIndexes.add(faceIndex)
                    s = set(adjacentFaces[faceIndex])
                    s &= regionDomain
                    s -= assignedIndexes
                    if s:
                        queue = UpdateQueue(region,
                                            faceNormals,
                                            areaFaces,
                                            queue,
                                            s)

    return regions

def BuildQueue(regions, faceNormals, areaFaces, adjacentToFaces):
    assignedIndexes = []
    queue = []
    for region in regions:
        seedIndex = region.faceIndexes[0]
        assignedIndexes.append(seedIndex)
        seedLocality = []
        seedLocality.extend(adjacentToFaces[seedIndex])
        queue = UpdateQueue(region,
                            faceNormals,
                            areaFaces,
                            queue,
                            seedLocality)
    return queue, assignedIndexes

def SplitRegion(faceNormals, areaFaces, adjacentFaces, regions, worst):
    worstRegion = FindRegion(regions,worst.regionIndex)
    splitRegion_A = generateId()
    spiltRegion_B = generateId()

    oldRegionFaces = worstRegion.faceIndexes
    seedIndex_A = oldRegionFaces[0]
    seedIndex_B = worst.index
    splitRegions = GetProxy([Proxy(splitRegion_A, [seedIndex_A]),Proxy(spiltRegion_B, [seedIndex_B])])

    queue, assignedIndexes = BuildQueue(splitRegions,
                                        faceNormals,
                                        areaFaces,
                                        adjacentFaces)

    splitRegions = AssignToWorstRegion(faceNormals,
                                       areaFaces,
                                       adjacentFaces,
                                       splitRegions,
                                       queue,
                                       assignedIndexes,
                                       oldRegionFaces)
    splitRegions[0].combinaisons = worst.regionIndex
    splitRegions[1].combinaisons = worst.regionIndex
    return splitRegions

def FindAdjacentRegions(faceEdges, regions):
    adjacentRegions = []
    regionsEdges = []
    for region in regions:
        regionIndex = region.regionIndex
        regionEdges = []
        for i in region.faceIndexes:
            regionEdges.extend(faceEdges[i])
        regionsEdges.append([regionIndex, set(regionEdges)])

    for region_A, region_B in itertools.combinations(regionsEdges, 2):
        commonEdges = set(region_A.faceIndexes).intersection(set(region_B.faceIndexes))
        if commonEdges:
            adjacentRegions.append([region_A.regionIndex, region_B.regionIndex])

    return adjacentRegions

def FindRegionsToCombine(regions, adjacentRegions, faceNormals, areaFaces):
    maxError = -Infinity
    regionsToCombine = None
    for i, adjacent in enumerate(adjacentRegions):
        region_A = regions[adjacent[0]]
        region_B = regions[adjacent[1]]
        mergedRegion = GetProxy([Proxy(generateId(), region_A.faceIndexes + region_B.faceIndexes, combinaisons=adjacent)])[0]
        regionError = 0
        proxyNormal = mergedRegion.proxyNormal
        for index in mergedRegion.faceIndexes:
            area = areaFaces[index]
            normal = faceNormals[index]
            try:
                normalError = normal - proxyNormal
            except TypeError:
                # proxyNormal = rs.VectorCreate([0,0,0], [0,0,0]) # CHANGé -> crée un vecteur à partir de 2 points : ici on initialise à un vecteur nul
                proxyNormal = np.array([0,0,0])
                normalError = normal - proxyNormal
            # moduleNormalError = normalError.SquareLength # CHANGé -> équivalent à la norme du vecteur au carré
            moduleNormalError = (np.linalg.norm(normalError)) ** 2
            regionError += moduleNormalError * area
            if regionError > maxError and i > 0:
                break
            else:
                regionsToCombine = mergedRegion
                maxError = regionError

    return regionsToCombine

def GrowSeeds(subFaceIndexes, faceVertexIndexes, vertices, color = None ):
    if not color:
        color = Randomcolor()
    t = [faceVertexIndexes[i] for i in subFaceIndexes]
    r =  set([i for sublist in t for i in sublist])
    mapa = dict(list(zip(r, list(range(len(r))))))
    subVertices = list(mapa.keys())
    newFaceIndexes = []
    for item in t:
        newFaceIndexes.append([mapa[i] for i in item])
    newVertices = {}
    for k, v in mapa.items():
        newVertices[v] = vertices[k]
    newVertices = list(newVertices.values())
    newMesh = Mesh(newVertices, newFaceIndexes)
    newId = generateId()
    ps.register_surface_mesh(newId, newVertices, newFaceIndexes, color=color)
    return newId, mapa, newMesh

def generateId():
    global ID_MESH_LAST
    ID_MESH_LAST += 1
    return ID_MESH_LAST

def Randomcolor():
    return random.randint(0, 255) / 255, random.randint(0, 255) / 255, random.randint(0, 255) / 255

def main():
    ps.init()

    # # sommets
    # verts=np.array([[1.,0.,0.],[0.,1.,0.],[-1.,0.,0.],[0.,-1.,0.],[0.,0.,1.]])
    # # faces (décrit à quels sommets la face est reliée)
    # faces=[[0,1,2,3],[1,0,4],[2,1,4],[3,2,4],[0,3,4]]
    # # maille
    # ps.register_surface_mesh("my mesh", verts, faces )
    
    obj = load_obj( 'helmet.obj')
    ps_mesh = ps.register_surface_mesh("helmet", obj.only_coordinates(), obj.only_faces() )
    mesh = Mesh(obj.only_coordinates(), obj.only_faces())
    print(mesh.getMeshAreaCentroid())
    # GetMeshAreaCentroid("my mesh")
    ps.show()

if __name__ == '__main__':
    main()