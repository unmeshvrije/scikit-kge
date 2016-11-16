import pickle, pprint
import sys

def main(datafile, e):
    with open(datafile,'rb')as fin:
        data = pickle.load(fin)

    train = data['train_subs']
    count = 0
    ed = int(e)
    for t in train:
        if t[0] == ed or t[1] == ed:
            count += 1;
    
    print count

if __name__=='__main__':
    if len(sys.argv) != 3:
        print "Pickle database file and entity ID must be given as an argument."
        sys.exit()
    main(sys.argv[1], sys.argv[2])

