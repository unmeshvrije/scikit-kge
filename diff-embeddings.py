import pickle, pprint, math
import sys
import pdb
from collections import defaultdict as ddict
import operator
import numpy
import operator

def processFile(datafile):
    with open(datafile,'r')as fin:
        data = fin.read()

    records = data.split(']')
    # Remove the last element (extra newline)
    del(records[-1])
    embeddings = [[] for _ in range(len(records))]
    for i,r in enumerate(records):
        embeddings_str = r.split(',[')[1].split()
        for e in embeddings_str:
            embeddings[i].append(float(e))

    return numpy.array(embeddings)
    #for i,e in enumerate(embeddings):
    #    print ("%d,%s\n" % (i, str(e)))

def processPickleFile(datafile):
    with open(datafile, 'rb') as fin:
        embeddings = pickle.load(fin)
    return embeddings

def l1Distance(em1, em2):
    distances = []
    for i, (e1, e2) in enumerate(zip(em1, em2)):
        out = 0
        r = numpy.abs(e1 - e2)
        for el in r:
            out += el
        distances.append((i, out))
    return distances

# cosine similarity function
# http://stackoverflow.com/questions/1746501/can-someone-give-an-example-of-cosine-similarity-in-a-very-simple-graphical-wa
def cosTheta(v1, v2):
    dot_product = sum(n1 * n2 for n1, n2 in zip(v1, v2) )
    magnitude1 = math.sqrt(sum(n ** 2 for n in v1))
    magnitude2 = math.sqrt(sum(n ** 2 for n in v2))
    return dot_product / (magnitude1 * magnitude2)


def similarity(em1, em2):
    out = []
    cos_dict = ddict()
    for i, (e1, e2) in enumerate(zip(em1, em2)):
        theta = cosTheta(e1, e2)
        cos_dict[i] = theta
        #print ("%d,%f" % (i, theta))

    sorted_dict = sorted(cos_dict.items(), key=operator.itemgetter(1))
    for k,v in enumerate(sorted_dict):
        out.append(v)
    return out

if __name__=='__main__':
    if len(sys.argv) != 4:
        print ("Embeddings files must be given as an arguments.")
        sys.exit()
    embeddings1 = processFile(sys.argv[1])
    embeddings2 = processFile(sys.argv[2])
    lubm1 = processPickleFile(sys.argv[3])
    if (len(embeddings1) != len(embeddings2)):
        print ("Both files contain different number of vectors")
        sys.exit()
    distances = similarity(embeddings1, embeddings2)
    distances = sorted(distances, key=operator.itemgetter(1))
    for d in distances:
        print(d, lubm1['entities'][d[0]])
    #similarity(embeddings1, embeddings2)

