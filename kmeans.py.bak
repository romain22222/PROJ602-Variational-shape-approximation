## kmeans.py by Jesus Galvez [jssglvz@gmail.com]
## This script will do kmeans clustering similar to [1] on a mesh on Python for 
## Rhinoceros [2].
##
## [1] David Cohen-Steiner, Pierre Alliez, and Mathieu Desbrun. Variational
##     shape approximation. ACM Transactions on Graphics, 23(3):905-
##     914, August 2004.
## [2] Robert McNeel & Associates. 2012. Rhinoceros: Release 5.0.

################################     LOG     ##################################

## 01
## added clustersMap as KMeans output.

#############################      END LOG     ################################

import Rhino
import rhinoscriptsyntax as rs
import rhinoscript.utility as rhutil
import scriptcontext

import random
import heapq
import time
import itertools

import sys
from System import DateTime
import gc


def KMeans(REDRAW, n, proxys, obj, faceCenters, faceNormals, faceCount, faceIndexes, vertices, faceVertexIndexes, areaFaces, faceEdges, weightedAverages, adjacentToFaces):

    startKMeans = DateTime.Now
    counter = 0
    colors = []
    #colors = [(157, 113, 233), (5, 35, 64), (4, 144, 116), (219, 69, 32), (75, 183, 53), (251, 161, 47), (212, 185, 239), (199, 128, 62), (18, 243, 188), (52, 36, 78), (104, 168, 195), (65, 116, 130), (165, 202, 65), (44, 21, 222), (183, 27, 128), (42, 175, 150), (195, 54, 201), (185, 61, 16), (192, 214, 240), (171, 243, 48), (131, 208, 107), (37, 115, 157), (93, 9, 67), (179, 85, 215), (237, 167, 163), (16, 15, 101), (190, 232, 68), (37, 112, 238), (133, 174, 106), (178, 129, 158), (145, 250, 214), (16, 168, 69), (115, 146, 167), (226, 204, 165), (83, 16, 224), (241, 204, 149), (109, 175, 198), (138, 237, 11), (236, 94, 95), (125, 159, 197), (50, 171, 136), (164, 34, 91), (0, 220, 255), (93, 157, 125), (190, 219, 27), (158, 210, 96), (13, 148, 226), (213, 68, 210), (215, 6, 12), (203, 36, 74), (230, 245, 242), (226, 176, 32), (29, 174, 238), (132, 240, 239), (150, 16, 17), (72, 204, 18), (239, 77, 139), (254, 111, 11), (227, 93, 103), (181, 205, 27)]
    for i in range(len(proxys)):
        color = Randomcolor(i)
        colors.append(color)
    
    for i in range(0, n):
        startCYCLE = DateTime.Now
        #print 'CYCLE ', counter
        ## This will calculate the mean of the face's barycenter and the mean
        ## of the face's normals weighted againts their face's area. This
        ## equals the proxy of the region. P = {X, N}, where X is the
        ## barycenter and N is the normal. It returns
        ## proxys = [[regionIndex,
        ##            [faceIndex],
        ##            proxyMesh,
        ##            proxyCenter,
        ##            proxyNormal,
        ##            proxyVector],
        ##          ...]
        proxys = GetProxy(proxys,
                          weightedAverages,
                          areaFaces)

        ## Get regions.
        ## This iterates through each face in a mesh trying to find which face
        ## has least metric error. The face with least error will be the seed
        ## face from which the region will grow.
        ## If counter = 0, region = proxy since there is only one face in each
        ## region.
        if counter > 0:
            ## This returns regions = [[regionIndex,
            ##                          [faceIndex],
            ##                          proxyMesh,
            ##                          proxyCenter,
            ##                          proxyNormal,
            ##                          proxyVector],
            ##                        ...]
            regions = GetProxySeed(proxys,
                                   faceNormals,
                                   areaFaces)
        else:
            regions = proxys

        ## First we need to fill the queue with the seed face's adjacent faces.
        ## regions = [[regionIndex,
        ##             [faceIndex],
        ##             proxyMesh,
        ##             proxyCenter,
        ##             proxyNormal,
        ##             proxyVector],
        ##           ...]
        queue, assignedIndexes = BuildQueue(regions,
                                            faceNormals,
                                            areaFaces,
                                            adjacentToFaces)

        ## Grow regions from seed faces according to the priority queue
        ## (metric error)
        ## Returned "worst" has structure = [error, regionIndex, index]
        ## regions is returned with [faceIndex] filled with the face indexes
        ## of that region. 
        regions, worst = AssignToRegion(faceNormals,
                                        vertices,
                                        faceVertexIndexes,
                                        areaFaces,
                                        adjacentToFaces,
                                        regions,
                                        queue,
                                        assignedIndexes)

        ## It's better to first split the worst region and then merge the two
        ## best ones.
        ## splitRegions =  [[len(regions),
        ##                   [faceIndex],
        ##                   [],
        ##                   [],
        ##                   proxyNormal,
        ##                   proxyVector,
        ##                   originalRegionIndex],
        ##                  [len(regions) + 1,
        ##                   [faceIndex],
        ##                   [],
        ##                   [],
        ##                   proxyNormal,
        ##                   proxyVector,
        ##                   originalRegionIndex]]
        splitRegions = SplitRegion(obj,
                                   faceCenters,
                                   faceNormals,
                                   vertices,
                                   faceVertexIndexes,
                                   areaFaces,
                                   adjacentToFaces,
                                   weightedAverages,
                                   regions,
                                   worst)

        ## Insert split regions in "regions".
        InsertRegions(regions,
                      splitRegions)

        ## This returns a list of list with the indexes of adjacent regions.
        ## adjacentRegions = [[regionIndex,
        ##                    adjacent region index],
        ##                   ...]
        adjacentRegions = FindAdjacentRegions(obj,
                                               faceEdges,
                                               regions)

        ## This returns regionsToCombine = [combinedRegion]
        ## regionsToCombine = [[region_A, region_B],
        ##                     [faceIndexes],
        ##                     [],
        ##                     [],
        ##                     proxyNormal]
        print adjacentRegions
        regionsToCombine = FindRegionsToCombine(regions,
                                                adjacentRegions,
                                                faceNormals,
                                                weightedAverages,
                                                areaFaces)

        ## Insert regionsToCombine in "regions".
        regions = InsertRegions(regions, [regionsToCombine])

        ## Delete old meshes, render new meshes, and create new proxys. 
        newProxys = []
        clustersMap = []
        for region in regions:
            rs.DeleteObjects(region[2])
            newProxyIndexes = region[1] 
            newProxyMesh, vertexMap = GrowSeeds(obj,
                                                faceCount,
                                                region[1],
                                                faceVertexIndexes,
                                                vertices,
                                                colors[region[0]])
            newProxys.append([region[0], 
                              newProxyIndexes,
                              newProxyMesh])
            clustersMap.append(vertexMap)
        proxys = newProxys

        ## This is used to enable real time rendering of the script. 
        if REDRAW:
            gc.collect()

        timeTaken = (DateTime.Now - startCYCLE).TotalMilliseconds
        print 'CYCLE ', counter, 'TOOK (times in milliseconds) = ', timeTaken
        #print '=================================================='

        counter += 1

        ###################### ITERATION END IN FOR LOOP ######################

    print '--------------------------------------------------'
    timeTaken = (DateTime.Now - startKMeans).TotalSeconds
    print "KMeans took %s seconds to finish" % timeTaken

    return proxys, clustersMap

################################## END MAIN ###################################











#/#############################################################################
#/                                 FUNCTIONS                                 ##
#/#############################################################################

def InsertRegions(regions, insertRegions):
    ## combinedRegion = [len(regions),
    ##                   [faceIndexes],
    ##                   [],
    ##                   proxyMesh,
    ##                   proxyNormal,
    ##                   proxyVector,
    ##                   region_A, region_B]
    ## splitRegions = [[{0, 1},
    ##                  [faceIndex],
    ##                  proxyMesh,
    ##                  proxyCenter,
    ##                  proxyNormal,
    ##                  proxyVector ,
    ##                  len(regions)+n],
    ##                ...]
    ## worst  = [error, regionIndex, index]

    remIndexes = set()

    for region in insertRegions:
        remIndexes.update(region.pop())

    ## Depending on the order it may invalidate the iterator, while popping
    ## items from regions.
    remIndexes = sorted(list(remIndexes), reverse = True)

    ## Delete meshes from regions to replace.
    for index in remIndexes:
        if regions[index][2]:
            rs.DeleteObject(regions[index][2])

    ## Pop regions to replace.
    for index in remIndexes:
        regions.pop(index)

    regions.extend(insertRegions)

    for index, region in enumerate(regions):
        region[0] = index

    return regions


def GetProxy(proxys, weightedAverages, areaFaces):
    ## This will append the proxy's center and normal to
    ## the proxy list.
    ## It returns proxys = [[regionIndex,
    ##                       [faceIndex],
    ##                       proxyMesh,
    ##                       proxyCenter,
    ##                       proxyNormal,
    ##                       proxyVector],
    ##                     ...]
    for proxy in proxys:
        try:
            indexes, mesh  = proxy[1], proxy[2]
            proxyCenter = (rs.MeshAreaCentroid(mesh))
            #print "The faces's center coordinate is (", proxyCenters, ')'
            proxyNormal, proxyVector = GetProxyNormal(indexes, weightedAverages)
            #print "The proxy's normal vector is (", proxyNormals, ')'
            proxy.append(proxyCenter)
            proxy.append(proxyNormal)
        except IndexError:
            ## It returns proxys = [[regionIndex,
            ##                       [faceIndex],
            ##                       [],
            ##                       [],
            ##                       proxyNormal],
            ##                     ...]
            indexes  = proxy[1]
            proxyNormal, proxyVector = GetProxyNormal(indexes, weightedAverages)
            #print "The proxy's normal vector is (", proxyNormals, ')'
            proxy.extend([ [], [] ])        ## This are placeholders
            proxy.append(proxyNormal)

    return proxys


def GetProxyNormal(indexes, weightedAverages):
    ## For a region R i , the optimal proxy normal Ni is simply
    ## equal to the vector:
    ## Sumation { | T i | * ni }, for every T i and n i in the region,
    ## after normalization to make it unit, where T i and n i equal
    ## the face's area and normal.

    proxyNormal = rs.VectorCreate([0,0,0], [0,0,0])

    for index in indexes:
        weightedAverage = weightedAverages[index]
        proxyNormal = rs.VectorAdd(proxyNormal, weightedAverage)

    proxyNormal, proxyVector = rs.VectorUnitize(proxyNormal), proxyNormal

    return (proxyNormal, proxyVector)


def GetProxySeed(proxys, faceNormals, areaFaces):
    ## This iterates through each face in a mesh trying to find
    ## which face has least metric error. The face with least
    ## error will be the seed face from which the region will grow.
    ## It returns regions = [[regionIndex,
    ##                        [faceIndex],
    ##                        proxyMesh,
    ##                        proxyCenter,
    ##                        proxyNormal,
    ##                        proxyVector],
    ##                      ...]

    regions = []
    for proxy in proxys:
        regionIndex = proxy[0]
        faceIndexes = proxy[1]
        proxyNormal = proxy[4]

        ## This returns a list errors = [[error, regionIndex, index]]
        errors = MetricError(regionIndex,
                             faceIndexes,
                             faceNormals,
                             areaFaces,
                             proxyNormal)
        errors = sorted(errors,
                        reverse=True)

        ## This returns regions = [[regionIndex,
        ##                          [faceIndex],
        ##                          proxyMesh,
        ##                          proxyCenter,
        ##                          proxyNormal],
        ##                        ...]
        seedFaceIndex = errors.pop()[2]
        region = [proxy[0],
                  [seedFaceIndex],
                  proxy[2],
                  proxy[3],
                  proxy[4]]
        regions.append(region)

    return regions


def MetricError(regionIndex, faceIndexes, faceNormals, areaFaces, proxyNormal):
    errors = []

    for index in faceIndexes:
            area = areaFaces[index]
            normal = faceNormals[index]
            try:
                normalError = normal - proxyNormal
            ## When calculating the proxyNormal if it's equal to
            ## [0,0,0] the result will instead be None. In this case
            ## the above operation will give a TypeError.
            except TypeError:
                proxyNormal = rs.VectorCreate([0,0,0], [0,0,0])
                normalError = normal - proxyNormal
            moduleNormalError = normalError.SquareLength
            error = moduleNormalError * area
            errors.append((error, regionIndex, index))

    ## This returns a list errors = [[error, regionIndex, index]]
    return errors


def UpdateQueue(region, faceNormals, areaFaces, queue, newFaces):
    ## This will calculate the error metric according to L 2,1.
    ## For a triangle T i of area | T i | , of normal ni , and of
    ## associated proxy P i = (X i , Ni ), the L 2,1 error is computed
    ## as follows:
    ## L 2,1 (T i , P ) = || ni - Ni || ^ 2 * | T i |
    regionIndex = region[0]
    proxyNormal = region[4]

    newFacesErrors = MetricError(regionIndex,
                                 newFaces,
                                 faceNormals,
                                 areaFaces,
                                 proxyNormal)
    queue.extend(newFacesErrors)
    ## This appears faster than queue.sort
    queue = sorted(queue,
                   reverse = True)
    #queue.sort(reverse = True)
    ## This returns a list queue = [[error, regionIndex, index], [...]]
    return queue


def UpdateQueueNew(region, faceNormals, areaFaces, queue, newFaces):
    ## This will calculate the error metric according to L 2,1.
    ## For a triangle T i of area | T i | , of normal ni , and of
    ## associated proxy P i = (X i , Ni ), the L 2,1 error is computed
    ## as follows:
    ## L 2,1 (T i , P ) = || ni - Ni || ^ 2 * | T i |
    regionIndex = region[0]
    proxyNormal = region[4]

    for index in newFaces:
        area = areaFaces[index]
        normal = faceNormals[index]
        try:
            normalError = normal - proxyNormal
        ## When calculating the proxyNormal if it's equal to
        ## [0,0,0] the result will instead be None. In this case
        ## the above operation will give a TypeError.
        except TypeError:
            proxyNormal = rs.VectorCreate([0,0,0], [0,0,0])
            normalError = normal - proxyNormal
        moduleNormalError = normalError.SquareLength
        error = moduleNormalError * area
        heapq.heappush(queue, (error, regionIndex, index))

    ## This returns a list queue = [(error, regionIndex, index), (...)]
    return queue


def AssignToRegion(faceNormals, vertices, faceVertexIndexes, areaFaces, adjacentFaces, regions, queue, assignedIndexes):
    ## Container list for the items popped from the priority list.
    heapq.heapify(queue)
    globalQueue = []
    assignedIndexes = set(assignedIndexes)
    ## Until the priority queue is not empty, keep popping
    ## the item with least priority from the priority queue.
    while queue:
            mostPriority = heapq.heappop(queue)
            faceIndex =  mostPriority[2]
            ## If the index of the popped face has already
            ## been assigned skip to the next one.
            if faceIndex not in assignedIndexes:
                globalQueue.append(mostPriority)
                regionIndex = mostPriority[1] ## regionIndex is Int
                regions[regionIndex][1].append(faceIndex)
                assignedIndexes.add(faceIndex)
                ## Get the adjacent faces of the popped face
                ## and append them to the priority queue.
                newAdjacentFaces = set(adjacentFaces[faceIndex])
                ## If an adjacent face has already been assigned
                ## to a region, skip it.
                newAdjacentFaces -= assignedIndexes
                ## Append faces to priority queue.
                queue = UpdateQueueNew(regions[regionIndex],
                                       faceNormals,
                                       areaFaces,
                                       queue,
                                       newAdjacentFaces)

    ## This will get the last element (largest error)
    globalQueue = sorted(globalQueue)
    ## This will pop the last element (largest error)
    worst = globalQueue.pop()


    ## This is incorrect because the last item fetched from the
    ## queue might not be the worst one, because it might have
    ## already been assigned to a region.
    #worst = mostPriority

    return (regions, worst)


def AssignToWorstRegion(faceNormals, vertices, faceVertexIndexes, areaFaces, adjacentFaces, regions, queue, assignedIndexes, oldRegionFaces):
    ## queue = [[error, regionIndex, index] , [...]]
    ## Container list for the faces in the old region.
    ## This equals the new regions domain.
    regionDomain = frozenset(oldRegionFaces)
    assignedIndexes = set(assignedIndexes)
    ## This will pop any faces that have been placed
    ## into the queue at BuildQueue which are not in
    ## the domain.
    queue = [i for i in queue if i[2] in regionDomain]
    ## Until the priority queue is not empty, keep popping
    ## the item with least priority from the priority queue.
    while queue :
        mostPriority = queue.pop()
        faceIndex = mostPriority[2]
        ## If the index of the popped face has already
        ## been assigned skip to the next one.
        if faceIndex not in assignedIndexes:
            regionIndex = mostPriority[1]  ## regionIndex is Int
            for region in regions:
                if regionIndex == region[0]:
                    region[1].append(faceIndex)
                    assignedIndexes.add(faceIndex)
                    ## Get the adjacent faces of the popped face
                    ## and append them to the priority queue.
                    s = set(adjacentFaces[faceIndex])
                    ## If an adjacent face has already been assigned
                    ## to a region of if it's outside the domain, skip it.
                    s &= regionDomain
                    s -= assignedIndexes
                    ## Append faces to priority queue.
                    if s:
                        queue = UpdateQueue(region,
                                            faceNormals,
                                            areaFaces,
                                            queue,
                                            s)

    return (regions)


def BuildQueue(regions, faceNormals, areaFaces, adjacentToFaces):
    ## Build priority queue.
    ## This will add the metric error of the seed face's adjacent
    ## to the queue and assign the seed indexes to [assignedIndexes]

    ## Container list for the indexes of the faces that have already
    ## been assigned to a region.
    assignedIndexes = []
    ## Container list for the priority list. It has structure:
    ## [[error, regionIndex, index]]
    queue = []

    ## This gets the metric error of the seed's adjacent faces,
    ## and adds it to the error queue.
    for region in regions:
        ## Get metric error.
        seedIndex = region[1][0] ## seedIndex = Int

        ## This will add the seed indexes to the assigned
        ## indexes list.
        assignedIndexes.append(seedIndex)

        seedLocality = []
        ## This can be omited since the seed index will not be
        ## appended to the global queue because it won't
        ## pass the test at AssignToRegion:
        ## if faceIndex not in assignedIndexes:
        #seedLocality.append(seedIndex)
        seedLocality.extend(adjacentToFaces[seedIndex])
        queue = UpdateQueue(region,
                            faceNormals,
                            areaFaces,
                            queue,
                            seedLocality)

    return (queue, assignedIndexes)


def SplitRegion(mesh, faceCenters, faceNormals, vertices, faceVertexIndexes, areaFaces, adjacentFaces, weightedAverages, regions, worst):
    ## worst = [error,
    ##          regionIndex,
    ##          index]
    ## Get worst region from list of regions.
    ## worstRegion = [regionIndex,
    ##                [faceIndex],
    ##                proxyMesh,
    ##                proxyCenter,
    ##                proxyNormal]
    worstRegion = regions[worst[1]]
    ## Get split region indexes.
    splitRegion_A = len(regions)
    spiltRegion_B = len(regions)+1
    ## Get region's face indexes.
    oldRegionFaces = worstRegion[1]      ## This is a list.
    ## Get worst region's seed indexes.
    seedIndex_A = oldRegionFaces[0]      ## This is an int.
    seedIndex_B = worst[2]               ## This is an int.
    ## Get proxy data for new seeds.
    splitRegions = [[splitRegion_A, [seedIndex_A]],
                    [spiltRegion_B, [seedIndex_B]]]
    ## It returns proxys = [[regionIndex,
    ##                       [faceIndex],
    ##                       [],
    ##                       [],
    ##                       proxyNormal,
    ##                       proxyVector],
    ##                     ...]
    splitRegions = GetProxy(splitRegions,
                            weightedAverages,
                            areaFaces)

    ## First we need to fill the queue with the seed face's adjacent faces.
    ## Regions have been modified with structure:
    ## splitRegions = [[regionIndex,
    ##                  [faceIndex],
    ##                  [],
    ##                  [],
    ##                  proxyNormal,
    ##                  proxyVector],
    ##                ...]
    queue, assignedIndexes = BuildQueue(splitRegions,
                                        faceNormals,
                                        areaFaces,
                                        adjacentFaces)

    ## Grow regions from seed faces according to the
    ## priority queue (metric error)
    ## Returned worst has structure = [error, regionIndex, index]
    splitRegions = AssignToWorstRegion(faceNormals,
                                       vertices,
                                       faceVertexIndexes,
                                       areaFaces,
                                       adjacentFaces,
                                       splitRegions,
                                       queue,
                                       assignedIndexes,
                                       oldRegionFaces)
    splitRegions[0].append([worstRegion[0]])
    splitRegions[1].append([worstRegion[0]])
    ## splitRegions =  [[len(regions),
    ##                   [faceIndex],
    ##                   [],
    ##                   [],
    ##                   proxyNormal,
    ##                   proxyVector,
    ##                   [originalRegionIndex]],
    ##                  [len(regions) + 1,
    ##                   [faceIndex],
    ##                   [],
    ##                   [],
    ##                   proxyNormal,
    ##                   proxyVector,
    ##                   [originalRegionIndex]]]
    return splitRegions


def FindAdjacentRegions(mesh_id, faceEdges, regions, addCommonEdges=False):
    ## This returns a list with the adjacent regions with structure:
    ## adjacentRegions = [[regionIndex, adjacent region index], ...]

    adjacentRegions = []
    regionsEdges = []

    for region in regions:
        regionIndex = region[0]
        regionEdges = []
        for i in region[1]:
            regionEdges.extend (faceEdges[i])
        regionsEdges.append([regionIndex, set(regionEdges)])

    for region_A, region_B in itertools.combinations(regionsEdges, 2):
        #print region_A[0], region_B[0]
        #print region_A[1], region_B[1]
        commonEdges = set (region_A [1]).intersection(set (region_B [1]))
        #print region_A[0], region_B[0], ':  Common Edges = ', commonEdges
        if commonEdges and (not addCommonEdges):
            adjacentRegions.append([region_A[0], region_B[0]])
        elif commonEdges and addCommonEdges:
            adjacentRegions.append([region_A[0], region_B[0], commonEdges])
    #for adjacent in adjacentRegions:
    #    print adjacent
    #print 'Adjacent Regions = ', adjacentRegions

    return adjacentRegions


def FindRegionsToCombine(regions, adjacentRegions, faceNormals, weightedAverages, areaFaces):
    ## This will find the two regions, that when combined have the least metric
    ## error.
    ## adjacentRegions = [[regionIndex, adjacent region index], ...]

    for i, adjacent in enumerate(adjacentRegions):
        ## This iterates through adjacentRegions merging the regions of index
        ## adjacentRegions[0] and adjacentRegions[1]. Merged regions are called
        ## mergedRegion. The first iteration calculates the error of all faces
        ## in that region. The remaining iterations only compute if the error
        ## is larger.

        if i > 0:
            ## This gets neighbouring region indexes.
            region_A = regions[adjacent[0]]
            region_B = regions[adjacent[1]]
            ## This creates a new region of region index equal to a list of
            ## both regions indexes, and with the combined face indexes of
            ## both regions.
            ## Note. adjacent is a list: [region_A_Index, region_B_Index]
            mergedRegion = [adjacent, region_A[1] + region_B[1]]
            ## This will get mergedRegion's proxy. It returns:
            ## mergedRegion = [[regionIndexes,        ## regionIndex = adjacent
            ##                 [faceIndex],
            ##                 [],
            ##                 [],
            ##                 proxyNormal]]
            mergedRegion = GetProxy([mergedRegion],   ## This is a list of list
                                 weightedAverages,
                                 areaFaces)
            mergedRegion = mergedRegion[0]            ## This is a list
            regionError = 0
            proxyNormal = mergedRegion[4]

            for index in mergedRegion[1]:
                area = areaFaces[index]
                normal = faceNormals[index]
                try:
                    normalError = normal - proxyNormal
                ## When calculating the proxyNormal if it's equal to
                ## [0,0,0] the result will instead be None. In this case
                ## the above operation will give a TypeError.
                except TypeError:
                    proxyNormal = rs.VectorCreate([0,0,0], [0,0,0])
                    normalError = normal - proxyNormal
                moduleNormalError = normalError.SquareLength
                regionError += moduleNormalError * area

                if regionError > maxError:
                    break
                else:
                    regionsToCombine = mergedRegion
                    maxError = regionError
        else:
            ## This gets neighbouring region indexes.
            region_A = regions[adjacent[0]]
            region_B = regions[adjacent[1]]
            ## This creates a new region with region index equal to a list of
            ## both regions indexes, and with the combined face indexes of
            ## both regions.
            ## Note. adjacent is a list: [region_A_Index, region_B_Index]
            mergedRegion = [adjacent, region_A[1] + region_B[1]]
            ## This will get mergedRegion's proxy. It returns:
            ## mergedRegion = [[regionIndexes,        ## regionIndex = adjacent
            ##                 [faceIndex],
            ##                 [],
            ##                 [],
            ##                 proxyNormal]]
            mergedRegion = GetProxy([mergedRegion],   ## This is a list of list
                                 weightedAverages,
                                 areaFaces)
            mergedRegion = mergedRegion[0]            ## This is a list
            regionError = 0
            proxyNormal = mergedRegion[4]

            for index in mergedRegion[1]:
                area = areaFaces[index]
                normal = faceNormals[index]
                try:
                    normalError = normal - proxyNormal
                ## When calculating the proxyNormal if it's equal to
                ## [0,0,0] the result will instead be None. In this case
                ## the above operation will give a TypeError.
                except TypeError:
                    proxyNormal = rs.VectorCreate([0,0,0], [0,0,0])
                    normalError = normal - proxyNormal
                moduleNormalError = normalError.SquareLength
                regionError += moduleNormalError * area
            regionsToCombine = mergedRegion
            maxError = regionError

    ## regionsToCombine = [regionIndexes,        ## regionIndex = adjacent
    ##                     [faceIndex],
    ##                     [],
    ##                     [],
    ##                     proxyNormal]
    AB_Index = regionsToCombine[0]
    regionsToCombine[0] = len(regions)
    regionsToCombine.append(AB_Index)

    ## This returns regionsCombine.
    ## regionsToCombine = [len(regions),
    ##                     [faceIndexes],
    ##                     [],
    ##                     [],
    ##                     proxyNormal,
    ##                     [region_A_Index, region_B_Index]]
    return regionsToCombine


def JoinMeshes(mesh_id_A, mesh_id_B, delete_input=False):
    object_ids = (mesh_id_A, mesh_id_B)

    mesh_A = rs.coercemesh(mesh_id_A, True)
    mesh_B = rs.coercemesh(mesh_id_B, True)

    try:
        mesh_A_Color = mesh_A.VertexColors[0]
        for i in xrange((mesh_B.VertexColors.Count)):
            mesh_B.VertexColors[i] = mesh_A_Color
    except IndexError:
        mesh_A.Append(mesh_B)
    rc = scriptcontext.doc.Objects.AddMesh(mesh_A)

    if delete_input:
        for id in object_ids:
            guid = id
            scriptcontext.doc.Objects.Delete(guid,True)

    return rc


def GrowSeeds(mesh, faceCount, subFaceIndexes, faceVertexIndexes, vertices, color = (155, 155, 155) ):
    ## This will create a new mesh from the indexes in subFaceIndexes. 
    ## Returns the guid of the new mesh. 
    ## subFaceIndexes is a list of indexes of the faces in the region, 
    ## referenced to the face indexes of (the original) mesh. Example [1, 3, 4]
    
    try: color
    except UnboundLocalError:
        color = Randomcolor(subFaceIndexes[0])

    ## t = vertex indexes of subFaceIndexes
    ## example: t = [(3, 1, 2, 2), (3, 2, 4, 4), (6, 3, 4, 4)]
    t = [faceVertexIndexes[i] for i in subFaceIndexes]
    #print 't = ', t
    ## r = t flattened
    ## example: r = [3, 1, 2, 2, 3, 2, 4, 4, 6, 3, 4, 4]
    r =  set([i for sublist in t for i in sublist])
    #r = [i for sublist in [faceVertexIndexes[i] for i in subFaceIndexes] for i in sublist]
    #print 'r = ', r

    ## mapa will map t onto new indexes starting from 0
    ## mapa {original mesh vertex index: new mesh vertex index}
    ## example: mapa {1: 0, 2: 1, 3: 2, 4: 3, 6: 4}
    mapa = dict(zip(r, range(len(r))))
    #print 'mapa = ' , mapa

    ## subVertices is a "set" of t.
    ## example: subVertices [1, 2, 3, 4, 6]
    subVertices = mapa.keys()
    #print 'subVertices = ', subVertices

    ## newFaceIndexes = vertex Indexes of subFaces mapped with mapa.
    ## example: [[2, 0, 1, 1], [2, 1, 3, 3], [4, 2, 3, 3]]
    newFaceIndexes = []
    for item in t:
        newFaceIndexes.append([mapa[i] for i in item])
    #print newFaceIndexes

    ## These are the coordinates of the vertices
    ## mapped onto their new indexes.
    newVertices = {}
    for k, v in mapa.iteritems():
        newVertices[v] = vertices[k]
    newVertices = list(newVertices.values())

    colors = [color for i in newVertices]

    return rs.AddMesh(newVertices, newFaceIndexes, vertex_colors=colors), mapa
                      


def Randomcolor(self):
    #colors = []

    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)

    colors = (r,g,b)

    return colors



#==============================================================================
#                                   END OF MODULE
#==============================================================================  