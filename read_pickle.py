import pickle, pprint
import sys

def main(datafile):
    with open(datafile,'rb')as fin:
        data = pickle.load(fin)

    print ("# (Entities) = %d" % (len(data['entities'])))
    print ("# (Relations) = %d" % (len(data['relations'])))
    print ("# (train_subs) = %d" % (len(data['train_subs'])))
    print ("# (test_subs) = %d" % (len(data['test_subs'])))
    print ("# (valid_subs) = %d" % (len(data['valid_subs'])))
    #pprint.pprint(data)

if __name__=='__main__':
    if len(sys.argv) != 2:
        print ("Pickle database file must be given as an argument.")
        sys.exit()
    main(sys.argv[1])

