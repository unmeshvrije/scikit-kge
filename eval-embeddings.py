import pickle, pprint, math
import sys
import pdb
from collections import defaultdict as ddict
import operator
import numpy
import operator
import time
import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('EVAL-EMBED')


def incoming_neighbours(entity, graph):
    relations_entity_is_tail = graph['incoming'][entity].keys()
    incoming_neighbours = []
    for r in relations_entity_is_tail:
        for e in graph['relations_tail'][r].keys():
            incoming_neighbours.append(e)

    return incoming_neighbours

def outgoing_neighbours(entity, graph):
    relations_entity_is_head = graph['outgoing'][entity].keys()
    outgoing_neighbours = []
    for r in relations_entity_is_head:
        for e in graph['relations_head'][r].keys():
            outgoing_neighbours.append(e)

    return outgoing_neighbours

def make_graph(triples, N, M):
    graph_outgoing = [ddict(list) for _ in range(N)]
    graph_incoming = [ddict(list) for _ in range(N)]
    graph_relations_head = [ddict(list)for _ in range(M)]
    graph_relations_tail = [ddict(list)for _ in range(M)]
    for t in triples:
        head = t[0]
        tail = t[1]
        relation = t[2]
        graph_outgoing[head][relation].append(tail)
        graph_incoming[tail][relation].append(head)
        graph_relations_head[relation][head].append(tail)
        graph_relations_tail[relation][tail].append(head)

    return {'outgoing': graph_outgoing, 'incoming': graph_incoming, 'relations_head': graph_relations_head, 'relations_tail':graph_relations_tail}

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

def processPickleFile(datafile):
    with open(datafile, 'rb') as fin:
        data = pickle.load(fin)
    return data

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


def cosineMatrix(em):
    N = len(em)
    mat = numpy.full((N,N), 2)

    for i in range(N):
        mat[i][i] = 1

    i = 0
    while (i < N):
        j = i+1
        while (j < N):
            mat[i][j] = mat[j][i] = cosTheta(em[i], em[j])
            j += 1
        i += 1

    return mat

def similarity(em, mat, TOPK, training_graph, test_graph, flog):
    out = [[] for i in range(len(em))]
    for i, e in enumerate(em):
        flog.write ("Entity (%d): " %(i))
        log.info ("Entity (%d): " %(i))
        cos_dict = ddict()
        #incoming = incoming_neighbours(i, graph)
        #outgoing = outgoing_neighbours(i, graph)
        incoming_train = incoming_neighbours(i, training_graph)
        outgoing_train = outgoing_neighbours(i, training_graph)
        incoming_test = incoming_neighbours(i, test_graph)
        outgoing_test = outgoing_neighbours(i, test_graph)
        for j, obj in enumerate(em):
            if i == j:
                continue
            theta = mat[i][j]
            cos_dict[j] = theta
        #print ("%d,%f" % (i, theta))

        sorted_dict = sorted(cos_dict.items(), key=operator.itemgetter(1), reverse=True)
        flog.write("cosine results collected and sorted, ")
        log.info("cosine results collected and sorted, ")

        # Add one more column for training/test/Unknown
        for k,v in enumerate(sorted_dict):
            if k == TOPK:
                break
            if k in incoming_train or k in outgoing_train:
                out[i].append((v, True, "TRAIN"))
            elif k in incoming_test or k in outgoing_test:
                out[i].append((v, True, "TEST"))
            else:
                out[i].append((v, False))
        flog.write(" Table of evaluation built\n")
        log.info(" Table of evaluation built\n")
    return out

if __name__=='__main__':
    if len(sys.argv) != 4:
        print ("Usage: python %s <embeddings.txt> <kb.bin> <TOPK>" % (sys.arg[0]))
        sys.exit()

    logfile = sys.argv[1] + ".log"
    flog = open(logfile, 'w')

    start = time.time()
    embeddings = processFile(sys.argv[1])
    kb = processPickleFile(sys.argv[2])
    flog.write("Time to process files  = %ds" % (time.time() - start))
    log.info("Time to process files  = %ds" % (time.time() - start))
    TOPK = int(sys.argv[3])
    N = len(kb['entities'])
    M = len(kb['relations'])
    training = kb['train_subs']
    valid = kb['valid_subs']
    test = kb['test_subs']
    # this is unfiltered evaluation (because the triples from the training sets are supposed to be guessed correctly)
    #dataset = training + valid + test
    start = time.time()
    training_graph = make_graph(training, N, M)
    test_graph = make_graph(test, N, M)
    flog.write("Time to build graphs from triples = %ds\n" %(time.time() - start))
    log.info("Time to build graphs from triples = %ds\n" %(time.time() - start))

    if N != len(embeddings):
        print("Number of entities don't match (embeddings file and database)")
        sys.exit()

    start = time.time()
    mat = cosineMatrix(embeddings)
    flog.write("Time to make cosine distance matrix = %ds\n" % (time.time() - start))
    log.info("Time to make cosine distance matrix = %ds\n" % (time.time() - start))

    start = time.time()
    cosines = similarity(embeddings, mat, TOPK, training_graph, test_graph, flog)
    flog.write("Time to sort and rank best matching objects = %ds\n"%(time.time() - start))
    log.info("Time to sort and rank best matching objects = %ds\n"%(time.time() - start))

    start = time.time()
    outFile = sys.argv[2] + "-" + "TOP-" + str(TOPK) + ".eval.out"
    data = "{"
    for i, pairs in enumerate(cosines):
        data += str(i) + ": {"
        for p in pairs:
            data += str(p) + "\n"
        data += "}"
    data += "}"
    with open(outFile, 'w') as fout:
        fout.write(data)
    flog.write("Time to write out file = %ds" % (time.time() - start))
    log.info("Time to write out file = %ds" % (time.time() - start))
