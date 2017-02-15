import matplotlib.pyplot as plt
import sys
from collections import defaultdict as ddict

file1 = sys.argv[1]
file2 = sys.argv[2]
file3 = sys.argv[3]
file4 = sys.argv[4]


def processLogFile(logFile):
    x = []
    y = []
    d = ddict(lambda: {'hits': 0 , 'miss' : 0})
    with open(logFile, 'r') as f:
        lines = f.readlines()
        for l in lines:
            tokens = l.split()
            degree = int(tokens[0])
            result = int(tokens[1])
            x.append(degree)
            y.append(result)
            if result == 1:
                d[degree]['hits'] += 1
            elif result == 2:
                d[degree]['miss'] += 1
            else:
                print ("Second column must contain 1 or 2 : %s" % (l))
                sys.exit()
    return x,y,d

def writeHistogramDataFile(logFile, dictionary):
    data = "deg Hits Misses\n"
    for record in dictionary.items():
        data += str(record[0]) +  " " + str(record[1]['hits']) +  " " + str(record[1]['miss']) + "\n"
    with open(logFile + "-hist.dat", 'w') as fout:
        fout.write(data)

x1, y1, d1 = processLogFile(file1)
x2, y2, d2 = processLogFile(file2)
x3, y3, d3 = processLogFile(file3)
x4, y4, d4 = processLogFile(file4)

writeHistogramDataFile(file1, d1)
writeHistogramDataFile(file2, d2)
writeHistogramDataFile(file3, d3)
writeHistogramDataFile(file4, d4)

#plt.legend(['head-pred-unfiltered', 'head-pred-filtered', 'tail-pred-unfiltered', 'tail-pred-filtered'], loc='upper left')
