import pickle, pprint
import sys

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
        cntTraining = int((cntEdges * 99.8)/100)
        cntTest = int((cntEdges * 0.1) / 100)
        cntValid = cntTest

        testIndex = int(cntTraining + cntTest)

        print ("Total triples : %d\n" % (cntEdges))

        entities_set = set()
        entities_map = {}
        entities = "u'entities': ["
        relations = "u'relations': [u'related_to'],"
        test = "u'test_subs':["
        trains = "u'train_subs':["
        valid = "u'valid_subs':["

        counter = 0
        identifier = 0
        for index, pair in enumerate(lines):
            counter += 1
            if len(pair.split()) < 2:
                print ("Line [%d] does not represent an edge" % (index))
                continue
            
            fromNode = pair.split()[0]
            toNode = pair.split()[1]
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
            fromNode = pair.split()[0]
            toNode = pair.split()[1]

            fwalks.write(str(entities_map[fromNode]) + "," + str(entities_map[toNode]) + "\n")

            if index < cntTraining-1:
                trains += "(" + str(entities_map[fromNode]) + "," + str(entities_map[toNode]) + ",0),\n"
            elif index == cntTraining - 1:
                trains += "(" + str(entities_map[fromNode]) + "," + str(entities_map[toNode]) + ",0)],"
            elif index < testIndex-1:
                test += "(" + str(entities_map[fromNode]) + "," + str(entities_map[toNode]) + ",0),\n"
            elif index == int(testIndex) - 1:
                test += "(" + str(entities_map[fromNode]) + "," + str(entities_map[toNode]) + ",0)],"
            elif index < cntEdges-1:
                valid += "(" + str(entities_map[fromNode]) + "," + str(entities_map[toNode]) + ",0),\n"
            else: # index == cntEdges-1
                valid += "(" + str(entities_map[fromNode]) + "," + str(entities_map[toNode]) + ",0)]"

        for e in entities_set:
            entities += "u'" + str(e) + "'," + "\n"
        entities += "],"

    data = "{\n" + entities + relations + trains + test + valid + "}"
    with open(datafile +'.pkl','w') as fout:
        fout.write(data)


if __name__=='__main__':
    if len(sys.argv) != 2:
        print ("Snap file database (edge list) must be given as an argument.")
        sys.exit()
    main(sys.argv[1])
