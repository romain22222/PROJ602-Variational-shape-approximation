import polyscope as ps
import numpy as np
from wavefront import *
import random
import time
import itertools

import sys
ID_MESH_LAST = 0

def calculateAreaOfTriangularFace(vect1, vect2):
    return np.linalg.norm(
                np.array(
                    [
                        vect1[1]*vect2[2]-vect1[2]*vect2[1],
                        vect1[2]*vect2[0]-vect1[0]*vect2[2],
                        vect1[0]*vect2[1]-vect1[1]*vect2[0]
                        ]
                    )
                ) * .5

def findEdgeInCorr(edge, corres):
    for i in range(len(corres)):
        if edge[0] == corres[i][0] and edge[1] == corres[i][1] and edge[2] == corres[i][2]:
            return i
    return len(corres)

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
            center = (v1 + v2 + v3) / 3
            area = calculateAreaOfTriangularFace(v2-v1, v3-v1)
            meshVolume += area
            temp = center * area

        return temp / meshVolume

    def getAllFacesArea(self):
        areas = []
        for face in self.faces:
            areas.append(
                calculateAreaOfTriangularFace(
                    self.vertices[face[1]]-self.vertices[face[0]],
                    self.vertices[face[2]]-self.vertices[face[0]]
                )
            )
        return areas

    def getAllFacesNormals(self):
        normals = []
        for face in self.faces:
            U = self.vertices[face[1]]-self.vertices[face[0]]
            V = self.vertices[face[2]]-self.vertices[face[0]]
            normals.append(
                (
                    U[1] * V[2] - U[2] * V[1],
                    U[2] * V[0] - U[0] * V[2],
                    U[0] * V[1] - U[1] * V[0]
                )
            )
        return normals

    def getAllAdjacentFaces(self):
        ajdF = []
        # on ne peut pas faire une initialisation classique,
        # car la création d'un set se fait par référence
        for i in range(len(self.faces)):
            ajdF.append(set())
        for i in range(len(self.faces)):
            for j in range(i+1, len(self.faces)):
                if len([k for k in [0,1,2] if self.faces[i][k] in self.faces[j]]) > 1:
                    ajdF[i].add(j)
                    ajdF[j].add(i)
        return ajdF

    def getAllFaceEdges(self):
        faceEdges = []
        correspondance = []
        for face in self.faces:
            p1 = findEdgeInCorr(self.vertices[face[0]]-self.vertices[face[1]], correspondance)
            if p1 == len(correspondance):
                correspondance.append(self.vertices[face[0]]-self.vertices[face[1]])
            p2 = findEdgeInCorr(self.vertices[face[0]] - self.vertices[face[2]], correspondance)
            if p2 == len(correspondance):
                correspondance.append(self.vertices[face[0]] - self.vertices[face[2]])
            p3 = findEdgeInCorr(self.vertices[face[2]] - self.vertices[face[1]], correspondance)
            if p3 == len(correspondance):
                correspondance.append(self.vertices[face[2]] - self.vertices[face[1]])

            faceEdges.append([p1,p2,p3])
        return faceEdges

class Proxy:
    def __init__(self, regionIndex, faceIndexes, proxyNormal=None, polyMesh=None):
        self.regionIndex = regionIndex
        self.faceIndexes = faceIndexes
        self.proxyNormal = proxyNormal
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
            try:
                ps.remove_surface_mesh(region.polyMesh)
            except TypeError:
                pass
            newProxyIndexes = region.faceIndexes
            newPolyMeshName, vertexMap, newProxyMesh = GrowSeeds(region.faceIndexes,
                                                faceVertexIndexes,
                                                vertices)#,
                                                # colors[region.regionIndex])
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

def calculateNewElementsOfQueue(queue, regionIndex, faces, proxyNormal, areaFaces, faceNormals, isInFindRegionToCombine=False, paramsFindRegionToCombine=None):
    for index in faces:
        area = areaFaces[index]
        normal = faceNormals[index]
        try:
            normalError = normal - proxyNormal
        except TypeError:
            proxyNormal = np.array([0,0,0])
            normalError = normal - proxyNormal
        moduleNormalError = (np.linalg.norm(normalError)) ** 2
        error = moduleNormalError * area

        if isInFindRegionToCombine:
            if error > paramsFindRegionToCombine["maxError"] and paramsFindRegionToCombine["i"] > 0:
                break
            else:
                paramsFindRegionToCombine["regionsToCombine"] = paramsFindRegionToCombine["mergedRegion"]
                paramsFindRegionToCombine["maxError"] = error
        else:
            queue.append(QueueElement(error, regionIndex, index))
    return paramsFindRegionToCombine if isInFindRegionToCombine else queue

def InsertRegions(regions, insertRegions):
    regions.extend(insertRegions)
    return regions

def GetProxy(proxys):
    for proxy in proxys:
        proxy.proxyNormal = GetProxyNormal(proxy.faceIndexes)
    return proxys

def GetProxyNormal(indexes):
    proxyNormal = np.array([0,0,0])
    for index in indexes:
        proxyNormal = np.add(proxyNormal, 1)
    proxyNormal = proxyNormal / np.linalg.norm(proxyNormal)
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
                  proxyNormal=proxy.proxyNormal)
        regions.append(region)
    return regions

def MetricError(regionIndex, faceIndexes, faceNormals, areaFaces, proxyNormal):
    return calculateNewElementsOfQueue([], regionIndex, faceIndexes, proxyNormal, areaFaces, faceNormals)

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
    calculateNewElementsOfQueue(queue, regionIndex, newFaces, proxyNormal, areaFaces, faceNormals)
    return queue

def AssignToRegion(faceNormals, areaFaces, adjacentFaces, regions, queue, assignedIndexes):
    globalQueue = []
    assignedIndexes = set(assignedIndexes)
    while queue:
        mostPriority = queue.pop()
        faceIndex =  mostPriority.index
        if faceIndex not in assignedIndexes:
            globalQueue.append(mostPriority)
            regionIndex = mostPriority.regionIndex
            regions[regionIndex].faceIndexes.append(faceIndex)
            assignedIndexes.add(faceIndex)
            newAdjacentFaces = set(adjacentFaces[faceIndex])
            newAdjacentFaces -= assignedIndexes
            queue = UpdateQueueNew(FindRegion(regions, regionIndex).regionIndex,
                                   faceNormals,
                                   areaFaces,
                                   queue,
                                   newAdjacentFaces)

    globalQueue.sort(key=lambda x: -x.error)
    try:
        worst = globalQueue.pop()
    except IndexError:
        # Si tous les éléments sont assignés, pas de globalQueue remplie
        worst = QueueElement(0.0,regions[0].regionIndex,regions[0].faceIndexes[0])

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
        queue = UpdateQueue(region,
                            faceNormals,
                            areaFaces,
                            queue,
                            adjacentToFaces[seedIndex])
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
        if region_A[1].intersection(region_B[1]):
            adjacentRegions.append([region_A[0], region_B[0]])

    return adjacentRegions

def FindRegionsToCombine(regions, adjacentRegions, faceNormals, areaFaces):
    params = {
        "maxError":-np.inf,
        "regionsToCombine": None,
        "mergedRegion": None,
        "i": None
    }
    for i, adjacent in enumerate(adjacentRegions):
        region_A = FindRegion(regions,adjacent[0])
        region_B = FindRegion(regions,adjacent[1])
        mergedRegion = GetProxy([Proxy(generateId(), region_A.faceIndexes + region_B.faceIndexes)])[0]
        proxyNormal = mergedRegion.proxyNormal
        params = {
            "maxError": params["maxError"],
            "regionsToCombine": params["regionsToCombine"],
            "mergedRegion": mergedRegion,
            "i": i
        }
        calculateNewElementsOfQueue([], 0, mergedRegion.faceIndexes, proxyNormal, areaFaces, faceNormals, True, params)
    return params["regionsToCombine"]

def GrowSeeds(subFaceIndexes, faceVertexIndexes, vertices, color = None ):
    if not color:
        color = Randomcolor()
    verticesOfRegionByFace = [faceVertexIndexes[i] for i in subFaceIndexes]
    verticesOfRegion =  set([i for sublist in verticesOfRegionByFace for i in sublist])
    mapa = dict(list(zip(verticesOfRegion, list(range(len(verticesOfRegion))))))
    subVertices = list(mapa.keys())
    newFaceIndexes = []
    for item in verticesOfRegionByFace:
        newFaceIndexes.append([mapa[i] for i in item])
    newVertices = {}
    for k, v in mapa.items():
        newVertices[v] = vertices[k]
    newVertices = list(newVertices.values())
    newMesh = Mesh(newVertices, newFaceIndexes)
    newId = generateId()
    ps.register_surface_mesh(str(newId), np.array(newVertices), newFaceIndexes, color=color)
    return newId, mapa, newMesh

def generateId():
    global ID_MESH_LAST
    ID_MESH_LAST += 1
    return ID_MESH_LAST

def Randomcolor():
    return random.randint(0, 255) / 255, random.randint(0, 255) / 255, random.randint(0, 255) / 255

def main():
    ps.init()
    # # pyramide
    # verts=np.array([[1.,0.,0.],[0.,1.,0.],[-1.,0.,0.],[0.,-1.,0.],[0.,0.,1.]])
    # faces=[[0,1,2,3],[1,0,4],[2,1,4],[3,2,4],[0,3,4]]
    # mesh = Mesh(verts, faces)

    # # dé à 8 faces
    verts=np.array([[1.,0.,0.],[-1.,0.,0.],[0.,1.,0.],[0.,-1.,0.],[0.,0.,1.],[0.,0.,-1.]])
    faces=[[0,2,4],[0,2,5],[0,3,4],[0,3,5],[1,2,4],[1,2,5],[1,3,4],[1,3,5]]
    mesh = Mesh(verts, faces)

    # casque (attention lourd)
    # obj = load_obj( 'helmet.obj')
    # ps_mesh = ps.register_surface_mesh("helmet", obj.only_coordinates(), obj.only_faces() )
    # mesh = Mesh(obj.only_coordinates(), obj.only_faces())
    KMeans(
        10,
        [Proxy(generateId(),[i]) for i in range(len(mesh.faces))],
        mesh.getAllFacesNormals(),
        mesh.vertices,
        mesh.faces,
        mesh.getAllFacesArea(),
        mesh.getAllFaceEdges(),
        mesh.getAllAdjacentFaces()
    )
    ps.show()

    # print(mesh.getAllFacesArea())
    # print(mesh.getAllFacesNormals())
    # print(mesh.getAllAdjacentFaces())
    # print(mesh.getAllFaceEdges())


if __name__ == '__main__':
    main()