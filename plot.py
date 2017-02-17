import matplotlib.pyplot as plt
import sys
from collections import defaultdict as ddict

file1 = sys.argv[1]
file2 = sys.argv[2]
file3 = sys.argv[3]
file4 = sys.argv[4]


def getRange(range_dict, degree):
    for i, r in enumerate(range_dict):
        if r['from'] <= degree and degree < r['to']:
            return i
    # Did not match any ranges. Degree must be larger than 3000
    return i

def processLogFile(logFile):
    x = []
    y = []
    d = ddict(lambda: {'hits': 0 , 'miss' : 0})
    range_dict = [
        {'from' : 0, 'to' : 10, 'hits': 0, 'miss' : 0},
        {'from' : 10, 'to' : 20, 'hits': 0, 'miss' : 0},
        {'from' : 20, 'to' : 50, 'hits': 0, 'miss' : 0},
        {'from' : 50, 'to' : 100, 'hits': 0, 'miss' : 0},
        {'from' : 100, 'to' : 200, 'hits': 0, 'miss' : 0},
        {'from' : 200, 'to' : 400, 'hits': 0, 'miss' : 0},
        {'from' : 400, 'to' : 800, 'hits': 0, 'miss' : 0},
        {'from' : 800, 'to' : 1200, 'hits': 0, 'miss' : 0},
        {'from' : 1200, 'to' : 3000, 'hits': 0, 'miss' : 0},
        {'from' : 3000, 'to' : 10000, 'hits': 0, 'miss' : 0}
    ]
    with open(logFile, 'r') as f:
        lines = f.readlines()
        for l in lines:
            tokens = l.split()
            degree = int(tokens[0])
            result = int(tokens[1])
            x.append(degree)
            y.append(result)
            r = getRange(range_dict, degree)
            if result == 1:
                d[degree]['hits'] += 1
                range_dict[r]['hits'] += 1
            elif result == 2:
                d[degree]['miss'] += 1
                range_dict[r]['miss'] += 1
            else:
                print ("Second column must contain 1 or 2 : %s" % (l))
                sys.exit()
    return x,y,d,range_dict

#def writeHistogramDataFile(logFile, dictionary):
#    data = "deg Hits Misses\n"
#    for record in dictionary.items():
#        data += str(record[0]) +  " " + str(record[1]['hits']) +  " " + str(record[1]['miss']) + "\n"
#    with open(logFile + "-hist.dat", 'w') as fout:
#        fout.write(data)

def writeHistogramDataFile(logFile, range_array):
    data = "deg-range Hits Misses\n"
    for record in range_array:
        data += "[" + str(record['from']) +  "," + str(record['to']) + ") " + str(record['hits']) +  " " + str(record['miss']) + "\n"
    with open(logFile + "-hist.dat", 'w') as fout:
        fout.write(data)

x1, y1, d1, r1 = processLogFile(file1)
x2, y2, d2, r2 = processLogFile(file2)
x3, y3, d3, r3 = processLogFile(file3)
x4, y4, d4, r4 = processLogFile(file4)

writeHistogramDataFile(file1, r1)
writeHistogramDataFile(file2, r2)
writeHistogramDataFile(file3, r3)
writeHistogramDataFile(file4, r4)

#plt.legend(['head-pred-unfiltered', 'head-pred-filtered', 'tail-pred-unfiltered', 'tail-pred-filtered'], loc='upper left')
