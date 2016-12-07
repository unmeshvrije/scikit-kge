import pickle, pprint, numpy
import sys

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

def l1Distance(em1, em2):
    for i, (e1, e2) in enumerate(zip(em1, em2)):
        r = e1 - e2
        print ("%d,%s\n" % (i, r))

if __name__=='__main__':
    if len(sys.argv) != 3:
        print ("Embeddings files must be given as an arguments.")
        sys.exit()
    embeddings1 = processFile(sys.argv[1])
    embeddings2 = processFile(sys.argv[2])
    if (len(embeddings1) != len(embeddings2)):
        print ("Both files contain different number of vectors")
        sys.exit()
    l1Distance(embeddings1, embeddings2)

