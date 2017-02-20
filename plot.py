import matplotlib.pyplot as plt
import sys
from collections import defaultdict as ddict


def getRange(range_dict, degree):
    for i, r in enumerate(range_dict):
        if r['from'] <= degree and degree < r['to']:
            return i
    # Did not match any ranges. Degree must be larger than 3000
    return i

def processLogFile(logFile):
    x = []
    y = []
    d = ddict(lambda: {'hits': 0 , 'miss' : 0, 'relations' : 0})
    range_dict = [
        {'from' : 0, 'to' : 10, 'hits': 0, 'miss' : 0, 'relations' : 0},
        {'from' : 10, 'to' : 20, 'hits': 0, 'miss' : 0, 'relations' : 0},
        {'from' : 20, 'to' : 50, 'hits': 0, 'miss' : 0, 'relations' : 0},
        {'from' : 50, 'to' : 100, 'hits': 0, 'miss' : 0, 'relations' : 0},
        {'from' : 100, 'to' : 200, 'hits': 0, 'miss' : 0, 'relations' : 0},
        {'from' : 200, 'to' : 400, 'hits': 0, 'miss' : 0, 'relations' : 0},
        {'from' : 400, 'to' : 800, 'hits': 0, 'miss' : 0, 'relations' : 0},
        {'from' : 800, 'to' : 1200, 'hits': 0, 'miss' : 0, 'relations' : 0},
        {'from' : 1200, 'to' : 3000, 'hits': 0, 'miss' : 0, 'relations' : 0},
        {'from' : 3000, 'to' : 10000, 'hits': 0, 'miss' : 0, 'relations' : 0}
    ]
    with open(logFile, 'r') as f:
        lines = f.readlines()
        for l in lines:
            tokens = l.split()
            degree = float(tokens[0])
            result = int(tokens[1])
            numRelations = 0
            if (len(tokens) > 3):
                totalDegree = float(tokens[2])
                numRelations = int(tokens[3])
            x.append(degree)
            y.append(result)
            r = getRange(range_dict, degree)
            if result == 1:
                d[degree]['hits'] += 1
                d[degree]['relations'] += numRelations
                range_dict[r]['hits'] += 1
                range_dict[r]['relations'] += numRelations
            elif result == 2:
                d[degree]['miss'] += 1
                d[degree]['miss'] += numRelations
                range_dict[r]['miss'] += 1
                range_dict[r]['relations'] += numRelations
            else:
                print ("Second column must contain 1 or 2 : %s" % (l))
                sys.exit()
    return x,y,d,range_dict

def writeHistogramDataFile(logFile, dictionary):
    data = "deg Hits Misses Relations\n"
    for record in dictionary.items():
        data += str(record[0]) +  " " + str(record[1]['hits']) +  " " + str(record[1]['miss']) + " " + str(record[1]['relations']) + "\n"
    with open(logFile + "-hist.dat", 'w') as fout:
        fout.write(data)

def writeHistogramDataFileRange(logFile, range_array):
    data = "deg-range Hits Misses Relations\n"
    for record in range_array:
        data += "[" + str(record['from']) +  "," + str(record['to']) + ") " + str(record['hits']) +  " " + str(record['miss']) + " " + str(record['relations']) + "\n"
    with open(logFile + "-hist.dat", 'w') as fout:
        fout.write(data)

cluster = False
if (len(sys.argv) > 2):
    if sys.argv[2] == 'range':
        cluster = True
file1 = sys.argv[1]
x1, y1, d1, r1 = processLogFile(file1)

if cluster:
    writeHistogramDataFileRange(file1, r1)
else:
    writeHistogramDataFile(file1, d1)
