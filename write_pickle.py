import pickle, pprint
import sys

def main(datafile):
    f = open(datafile, 'r')
    data = eval(f.read())
    print (len(data['entities']))
    print ("%3d training tuples\n" % (len(data['train_subs'])))
    with open('pickle-db.pkl','wb')as fout:
        pickle.dump(data, fout, -1)


if __name__=='__main__':
    if len(sys.argv) != 2:
        print ("Pickle database file must be given as an argument.")
        sys.exit()
    main(sys.argv[1])
