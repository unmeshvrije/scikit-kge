from enum import Enum
import pickle
import copy
SUBTYPE = Enum('SUBTYPE', 'SPO POS')

class Metadata():
    def __init__(self, sid, st, sent, srel, ssize, entities):
        self.subType = st
        self.subId   = sid
        self.ent     = sent
        self.rel     = srel
        self.size    = ssize
        self.entities= copy.deepcopy(entities)
class Subgraphs():
    def __init__(self):
        self.subgraphs = []
    def add_subgraphs(self, st, sent, srel, ssize, entities):
        subentities = copy.deepcopy(entities)
        met = Metadata(len(self.subgraphs), st, sent, srel, ssize, subentities)
        self.subgraphs.append(met)
    def get_Nsubgraphs(self):
        return len(self.subgraphs)
    def save(self, fname, protocol=pickle.HIGHEST_PROTOCOL):
        with open(fname, 'wb') as fout:
            pickle.dump(self, fout, protocol=protocol)

    @staticmethod
    def load(fname):
        with open(fname, 'rb') as fin:
            subgraphs = pickle.load(fin)
        return subgraphs
