import pickle, pprint
import sys
from random import shuffle
#
#relations
#entities
#train_subs
#valid_subs
#test_subs
#

def main(datafile):
    with open(datafile, 'r') as f:
        lines = f.readlines()
        cntEdges = len(lines)
        if cntEdges < 100000:
            percentageTraining = 80
        else:
            percentageTraining = 99.8
        cntTraining = int((cntEdges * percentageTraining)/100)
        cntTest = int((cntEdges * (float(100 - percentageTraining)/2)) / 100)
        cntValid = cntTest

        testIndex = int(cntTraining + cntTest)

        print ("Total triples : %d\n" % (cntEdges))

        entities_set = set()
        relations_set = set()
        entities_map = {}
        relations_map = {}
        entities = "u'entities': ["
        relations = "u'relations':["
        test = "u'test_subs':["
        trains = "u'train_subs':["
        valid = "u'valid_subs':["

        counter = 0
        identifier = 0
        relationId = 0

        shuffle(lines)

        for index, pair in enumerate(lines):
            counter += 1
            if len(pair.split()) < 2:
                print ("Line [%d] does not represent an edge" % (index))
                continue
            parts = pair.split()
            fromNode = parts[0]
            toNode = parts[1]
            if (len(parts) > 2):
                edgeLabel = parts[2]
                if edgeLabel not in relations_set:
                    relations_set.add(edgeLabel)
                    relations_map[edgeLabel] = relationId
                    relationId +=1

            if fromNode not in entities_set:
                entities_set.add(fromNode)
                entities_map[fromNode] = identifier
                identifier += 1

            if toNode not in entities_set:
                entities_set.add(toNode)
                entities_map[toNode] = identifier
                identifier += 1

        with open(datafile + '.entity.map', 'w') as fen:
            fen.write(str(entities_map))

        fwalks = open(datafile + '.0-based-entitiy-ids.edgelist', 'w')
        print ("# of identifiers (entities) = %d" % (identifier))
        for index, pair in enumerate(lines):
            parts = pair.split()
            fromNode = parts[0]
            toNode = parts[1]
            edgeLabel = 0
            if len(parts) > 2:
                edgeLabel = parts[2]

            fwalks.write(str(entities_map[fromNode]) + " " + str(entities_map[toNode]) + "\n")

            if index < cntTraining-1:
                trains += "(" + str(entities_map[fromNode]) + "," + str(entities_map[toNode]) + ","+str(edgeLabel)+"),\n"
            elif index == cntTraining - 1:
                trains += "(" + str(entities_map[fromNode]) + "," + str(entities_map[toNode]) + ","+str(edgeLabel)+")],"
            elif index < testIndex-1:
                test += "(" + str(entities_map[fromNode]) + "," + str(entities_map[toNode]) + ","+str(edgeLabel)+"),\n"
            elif index == int(testIndex) - 1:
                test += "(" + str(entities_map[fromNode]) + "," + str(entities_map[toNode]) + ","+str(edgeLabel)+")],"
            elif index < cntEdges-1:
                valid += "(" + str(entities_map[fromNode]) + "," + str(entities_map[toNode]) + ","+str(edgeLabel)+"),\n"
            else: # index == cntEdges-1
                valid += "(" + str(entities_map[fromNode]) + "," + str(entities_map[toNode]) + ","+str(edgeLabel)+")]"

        for e in entities_set:
            entities += "u'" + str(e) + "'," + "\n"
        entities += "],"

        if len(relations_set) == 0:
            relations += "u'related_to', u'fake'],\n"
        else:
            for r in relations_set:
                relations += "u'" + str(r) + "'," + "\n"
            relations += "],"

    data = "{\n" + entities + relations + trains + test + valid + "}"
    with open(datafile +'.pkl','w') as fout:
        fout.write(data)


if __name__=='__main__':
    if len(sys.argv) != 2:
        print ("Snap file database (edge list) must be given as an argument.")
        sys.exit()
    main(sys.argv[1])
