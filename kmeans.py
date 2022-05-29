import polyscope as ps
import numpy as np
from wavefront import *
import random
import time
import itertools
import sys
import polyscope.imgui as psim

ID_MESH_LAST = 0

def calculateAreaOfTriangularFace(vect1, vect2):
    return np.linalg.norm(
        np.array(
            [
                vect1[1] * vect2[2] - vect1[2] * vect2[1],
                vect1[2] * vect2[0] - vect1[0] * vect2[2],
                vect1[0] * vect2[1] - vect1[1] * vect2[0]
            ]
        )
    ) * .5


def findEdgeInCorr(edge, corres):
    for i in range(len(corres)):
        if edge[0] == corres[i][1][0] and edge[1] == corres[i][1][1] and edge[2] == corres[i][1][2]:
            return i
    return len(corres)


class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces

    def getMeshAreaCentroid(self):
        meshVolume = 0
        temp = [0, 0, 0]

        for face in self.faces:
            v1 = self.vertices[face[0]]
            v2 = self.vertices[face[1]]
            v3 = self.vertices[face[2]]
            center = (v1 + v2 + v3) / 3
            area = calculateAreaOfTriangularFace(v2 - v1, v3 - v1)
            meshVolume += area
            temp = center * area

        return temp / meshVolume

    def getAllFacesArea(self):
        areas = []
        for face in self.faces:
            areas.append(
                calculateAreaOfTriangularFace(
                    self.vertices[face[1]] - self.vertices[face[0]],
                    self.vertices[face[2]] - self.vertices[face[0]]
                )
            )
        return areas

    def getAllFacesNormals(self):
        normals = []
        for face in self.faces:
            U = self.vertices[face[1]] - self.vertices[face[0]]
            V = self.vertices[face[2]] - self.vertices[face[0]]
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
            for j in range(i + 1, len(self.faces)):
                if len([k for k in [0, 1, 2] if self.faces[i][k] in self.faces[j]]) > 1:
                    ajdF[i].add(j)
                    ajdF[j].add(i)
        return ajdF

    def getAllFaceEdges(self):
        faceEdges = []
        correspondance = []
        toCheck = []
        # théorie : faire calcul x**3 + y**5 + z**7, on va obtenir une coordonée unique
        for face in self.faces:
            edges = calcEdges(self.vertices, face)
            faceE = []
            for i in range(3):
                p = calcUniqueCoordinate(edges[i])
                if p not in toCheck:
                    correspondance.append(p)
                    toCheck.append(p)
                else:
                    toCheck.remove(p)
                faceE.append(correspondance.index(p))
            faceEdges.append(faceE)

        return faceEdges


def calcUniqueCoordinate(edge):
    return edge[0] ** 3 + edge[1] ** 5 + edge[2] ** 7


def ordonne(vectice1, vectice2):
    if vectice1[0] != vectice2[0]:
        return vectice1[0] > vectice2[0]
    if vectice1[1] != vectice2[1]:
        return vectice1[1] > vectice2[1]
    return vectice1[2] > vectice2[2]


def calcEdges(vertices, face):
    edges = []
    for i in range(3):
        if ordonne(vertices[face[i]], vertices[face[i - 1]]):
            edges.append(vertices[face[i]] - vertices[face[i - 1]])
        else:
            edges.append(vertices[face[i - 1]] - vertices[face[i]])
    return edges


class Proxy:
    def __init__(self, regionIndex, faceIndexes, proxyNormal=None, vertices=None, faces=None):
        self.regionIndex = regionIndex
        self.faceIndexes = faceIndexes
        self.proxyNormal = proxyNormal
        self.polyMesh = None
        self.vertices = vertices
        self.faces = faces

    def draw(self, color):
        if self.polyMesh:
            self.undraw()
        self.polyMesh = generateId()
        ps.register_surface_mesh(self.polyMesh, self.vertices, self.faces, color=color)

    def undraw(self):
        try:
            if self.polyMesh:
                ps.remove_surface_mesh(self.polyMesh)
            else:
                raise TypeError
        except TypeError:
            pass


class QueueElement:
    def __init__(self, error, regionIndex, index):
        self.error = error
        self.regionIndex = regionIndex
        self.index = index


def KMeans(n, proxys, faceNormals, vertices, faceVertexIndexes, areaFaces, faceEdges, adjacentToFaces):
    for i in range(n):
        proxys = GetProxy(proxys)
        regions = GetProxySeed(proxys, faceNormals, areaFaces) if i > 0 else proxys
        queue, assignedIndexes = BuildQueue(regions, faceNormals, areaFaces, adjacentToFaces)
        regions, worst = AssignToRegion(faceNormals, areaFaces, adjacentToFaces, regions, queue, assignedIndexes)
        regions = SplitRegion(faceNormals, areaFaces, adjacentToFaces, regions, worst)
        adjacentRegions = FindAdjacentRegions(faceEdges, regions)
        regions = FindRegionsToCombine(regions, adjacentRegions, faceNormals, areaFaces)
        newProxys = []
        for region in regions:
            newProxyIndexes = region.faceIndexes
            verticesProxy, facesProxy = GrowSeeds(region.faceIndexes,
                                                  faceVertexIndexes,
                                                  vertices)
            newProxys.append(Proxy(region.regionIndex, newProxyIndexes, vertices=verticesProxy, faces=facesProxy))
        proxys = newProxys
    return proxys


def RefreshAllProxys(oldProxys, newProxys):
    for proxy in oldProxys:
        proxy.undraw()
    for proxy in newProxys:
        proxy.draw(Randomcolor())


def FindRegion(regions, index):
    return [region for region in regions if region.regionIndex == index][0]


def RemoveRegion(regions, regionIndex):
    region = FindRegion(regions, regionIndex)
    newRegions = [reg for reg in regions if reg.regionIndex != regionIndex]
    del region
    return newRegions


def calculateNewElementsOfQueue(queue, regionIndex, faces, proxyNormal, areaFaces, faceNormals,
                                isInFindRegionToCombine=False, paramsFindRegionToCombine=None):
    for index in faces:
        area = areaFaces[index]
        normal = faceNormals[index]
        try:
            normalError = normal - proxyNormal
        except TypeError:
            proxyNormal = np.array([0, 0, 0])
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
    proxyNormal = np.array([0, 0, 0])
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
    return calculateNewElementsOfQueue(queue, region.regionIndex, newFaces, region.proxyNormal, areaFaces, faceNormals)


def AssignToRegion(faceNormals, areaFaces, adjacentFaces, regions, queue, assignedIndexes):
    globalQueue = []
    assignedIndexes = set(assignedIndexes)
    while queue:
        mostPriority = queue.pop()
        faceIndex = mostPriority.index
        if faceIndex not in assignedIndexes:
            globalQueue.append(mostPriority)
            region = FindRegion(regions, mostPriority.regionIndex)
            region.faceIndexes.append(faceIndex)
            assignedIndexes.add(faceIndex)
            newAdjacentFaces = set(adjacentFaces[faceIndex])
            newAdjacentFaces -= assignedIndexes
            queue = UpdateQueueNew(region,
                                   faceNormals,
                                   areaFaces,
                                   queue,
                                   newAdjacentFaces)

    globalQueue.sort(key=lambda x: -x.error)
    try:
        worst = globalQueue.pop()
    except IndexError:
        # Si tous les éléments sont assignés, pas de globalQueue remplie
        worst = QueueElement(0.0, regions[0].regionIndex, regions[0].faceIndexes[0])

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
    worstRegion = FindRegion(regions, worst.regionIndex)
    splitRegion_A = generateId()
    spiltRegion_B = generateId()

    oldRegionFaces = worstRegion.faceIndexes
    seedIndex_A = oldRegionFaces[0]
    seedIndex_B = worst.index
    splitRegions = GetProxy([Proxy(splitRegion_A, [seedIndex_A]), Proxy(spiltRegion_B, [seedIndex_B])])

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
    return InsertRegions(RemoveRegion(regions, worstRegion.regionIndex), splitRegions)


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
        "maxError": -np.inf,
        "regionsToCombine": None,
        "mergedRegion": None,
        "i": None
    }
    regionsToDelete = adjacentRegions[0]
    for i, adjacent in enumerate(adjacentRegions):
        region_A = FindRegion(regions, adjacent[0])
        region_B = FindRegion(regions, adjacent[1])
        mergedRegion = GetProxy([Proxy(generateId(), region_A.faceIndexes + region_B.faceIndexes)])[0]
        proxyNormal = mergedRegion.proxyNormal
        params = {
            "maxError": params["maxError"],
            "regionsToCombine": params["regionsToCombine"],
            "mergedRegion": mergedRegion,
            "i": i
        }
        params = calculateNewElementsOfQueue([], 0, mergedRegion.faceIndexes, proxyNormal, areaFaces, faceNormals, True,
                                             params)
        if params["regionsToCombine"] == mergedRegion:
            regionsToDelete = adjacent
    return InsertRegions(RemoveRegion(RemoveRegion(regions, regionsToDelete[0]), regionsToDelete[1]),
                         [params["regionsToCombine"]])


def GrowSeeds(subFaceIndexes, faceVertexIndexes, vertices):
    verticesOfRegionByFace = [faceVertexIndexes[i] for i in subFaceIndexes]
    verticesOfRegion = set([i for sublist in verticesOfRegionByFace for i in sublist])
    mapa = dict(list(zip(verticesOfRegion, list(range(len(verticesOfRegion))))))
    subVertices = list(mapa.keys())
    newFaceIndexes = []
    for item in verticesOfRegionByFace:
        newFaceIndexes.append([mapa[i] for i in item])
    newVertices = {}
    for k, v in mapa.items():
        newVertices[v] = vertices[k]
    newVertices = list(newVertices.values())
    return np.array(newVertices), newFaceIndexes


def generateId():
    global ID_MESH_LAST
    ID_MESH_LAST += 1
    return str(ID_MESH_LAST)


def Randomcolor():
    return random.randint(0, 255) / 255, random.randint(0, 255) / 255, random.randint(0, 255) / 255


def generateNRegions(mesh, nb, adjacency):
    listFaces = mesh.faces[:]
    regions = []
    faceDrawn = []
    for i in range(nb):
        face = random.randrange(len(listFaces))
        while face in faceDrawn:
            face = random.randrange(len(listFaces))
        regions.append(Proxy(generateId(), [face]))
        faceDrawn.append(face)
    return regions

def corpse():
    global nbExec,vertsGlobal,facesGlobal,proxysGlobal,normalsGlobal,meshGlobal,areasGlobal,edgesGlobal,adjacencyGlobal
    psim.PushItemWidth(150)
    psim.TextUnformatted("Exécuter l'algorithme")
    psim.Separator()

    changed, nbExec = psim.InputInt("Nombre d'exécution de l'algo", nbExec, step=1, step_fast=3)
    psim.SameLine()
    if psim.Button("Exécuter"):
        newProxys = KMeans(
            nbExec,
            proxysGlobal,
            normalsGlobal,
            meshGlobal.vertices,
            meshGlobal.faces,
            areasGlobal,
            edgesGlobal,
            adjacencyGlobal
        )
        RefreshAllProxys(proxysGlobal, newProxys)
        proxysGlobal = newProxys

nbExec = 1
vertsGlobal = None
facesGlobal = None
proxysGlobal = None
normalsGlobal = None
meshGlobal = None
areasGlobal = None
edgesGlobal = None
adjacencyGlobal = None
def main():
    global vertsGlobal,facesGlobal,proxysGlobal,normalsGlobal,meshGlobal,areasGlobal,edgesGlobal,adjacencyGlobal
    choixFig = int(input("1 - Pyramide\n2 - Dé à 8 faces\n3 - Via un fichier obj\n"))
    if choixFig == 1:
        # pyramide
        vertsGlobal = np.array([[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
        facesGlobal = [[0, 1, 2, 3], [1, 0, 4], [2, 1, 4], [3, 2, 4], [0, 3, 4]]
    elif choixFig == 2:
        # dé à 8 faces
        vertsGlobal = np.array([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]])
        facesGlobal = [[0, 2, 4], [0, 2, 5], [0, 3, 4], [0, 3, 5], [1, 2, 4], [1, 2, 5], [1, 3, 4], [1, 3, 5]]
    else:
        nomObj = input("Entrez le nom du .obj (disponible normalement : 'helmet.obj'")
        obj = load_obj(nomObj)
        vertsGlobal = obj.only_coordinates()
        facesGlobal = obj.only_faces()
    ps.register_surface_mesh("MAIN", vertsGlobal, facesGlobal, color=(0., 1., 0.), edge_color=(0., 0., 0.), edge_width=3)
    meshGlobal = Mesh(vertsGlobal, facesGlobal)
    st = time.time()
    normalsGlobal = meshGlobal.getAllFacesNormals()
    print("normals : ", time.time() - st)
    st = time.time()
    areasGlobal = meshGlobal.getAllFacesArea()
    print("areas : ", time.time() - st)
    st = time.time()
    edgesGlobal = meshGlobal.getAllFaceEdges()
    print("edges : ", time.time() - st)
    st = time.time()
    adjacencyGlobal = meshGlobal.getAllAdjacentFaces()
    print("adjacency : ", time.time() - st)
    nbProxys = int(input("Combien de régions ?"))
    proxysGlobal = generateNRegions(meshGlobal, nbProxys, adjacencyGlobal)

    ps.init()
    ps.set_user_callback(corpse)
    ps.show()

    # print(mesh.getAllFacesArea())
    # print(mesh.getAllFacesNormals())
    # print(adjacency)
    # print(mesh.getAllFaceEdges())


if __name__ == '__main__':
    main()
