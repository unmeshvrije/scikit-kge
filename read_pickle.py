import pickle, pprint
import sys

def main(datafile):
    with open(datafile,'rb')as fin:
        data = pickle.load(fin)

    pprint.pprint(data)

if __name__=='__main__':
    if len(sys.argv) != 2:
        print ("Pickle database file must be given as an argument.")
        sys.exit()
    main(sys.argv[1])

