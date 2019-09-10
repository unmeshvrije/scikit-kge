from __future__ import print_function
import argparse
import numpy as np
from numpy import argsort
from numpy.random import shuffle
from collections import defaultdict as ddict
from skge.param import Parameter, AdaGrad
import timeit
import pickle
import pdb
import logging
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from skge import sample
from skge.util import to_tensor
import copy
import itertools
import sys
from skge.util import ccorr
from enum import Enum
from subgraphs import Subgraphs
import trident
SUBTYPE = Enum('SUBTYPE', 'SPO POS')

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('EX-KG')

_cutoff = 30

_DEF_NBATCHES = 100
_DEF_POST_EPOCH = []
_DEF_LEARNING_RATE = 0.1
_DEF_SAMPLE_FUN = None
_DEF_MAX_EPOCHS = 1000
_DEF_MARGIN = 1.0
_FILE_GRADIENTS = 'gradients.txt'
_FILE_EMBEDDINGS = 'embeddings.txt'
_FILE_TAIL_PREDICTIONS_UNFILTERED = 'tail-predictions-unfiltered.txt'
_FILE_TAIL_PREDICTIONS_FILTERED = 'tail-predictions-filtered.txt'
_FILE_HEAD_PREDICTIONS_UNFILTERED = 'head-predictions-unfiltered.txt'
_FILE_HEAD_PREDICTIONS_FILTERED = 'head-predictions-filtered.txt'
_FILE_INFO = 'info.txt'

_SIM_RANK_C = 0.6
_SIM_RANK_K = 5

np.random.seed(42)

graph = ddict()


def num_incoming_neighbours(entity, graph):
    all_lists = list(graph['incoming'][entity].values())
    incoming_neighbours = list(itertools.chain(*all_lists))
    return len(incoming_neighbours)

def num_outgoing_neighbours(entity, graph):
    all_lists = list(graph['outgoing'][entity].values())
    outgoing_neighbours = list(itertools.chain(*all_lists))
    return len(outgoing_neighbours)

def num_outgoing_relations(entity, graph):
    all_relations = list(graph['outgoing'][entity].keys())
    return len(all_relations)

def num_incoming_relations(entity, graph):
    all_relations = list(graph['incoming'][entity].keys())
    return len(all_relations)


def _is_converge(s1, s2, eps=1e-4):
    for i in s1.keys():
        for j in s1[i].keys():
            if abs(s1[i][j] - s2[i][j]) >= eps:
                return False
    return True





class Experiment(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser(prog='Knowledge Graph experiment', conflict_handler='resolve')
        self.parser.add_argument('--margin', type=float, help='Margin for loss function')
        self.parser.add_argument('--init', type=str, default='nunif', help='Initialization method')
        self.parser.add_argument('--lr', type=float, help='Learning rate')
        self.parser.add_argument('--me', type=int, help='Maximum number of epochs')
        self.parser.add_argument('--ne', type=int, help='Numer of negative examples', default=1)
        self.parser.add_argument('--nb', type=int, help='Number of batches')
        self.parser.add_argument('--fout', type=str, help='Path to store model and results', default=None)
        self.parser.add_argument('--finfo', type=str, help='Path to store additional debug info', default=None)
        self.parser.add_argument('--fgrad', type=str, help='Path to store gradient vector updates for each entity', default=None)
        self.parser.add_argument('--fpagerank', type=str, help='Path of the page ranks of all entities (in form of python dictionary)', default=None)
        self.parser.add_argument('--fembed', type=str, help='Path to store final embeddings for every entity and relation', default=None)
        self.parser.add_argument('--fin', type=str, help='Path to input data', default=None)
        self.parser.add_argument('--ftax', type=str, help='Path to the taxonomy file', default=None)
        self.parser.add_argument('--fsub', type=str, help='Path to the subgraphs file', default=None)
        self.parser.add_argument('--embed', type=str, help='Strategy to assign embeddings', default='kognac')
        self.parser.add_argument('--test-all', type=int, help='Evaluate Test set after x epochs', default=10)
        self.parser.add_argument('--no-pairwise', action='store_const', default=False, const=True)
        self.parser.add_argument('--incr', type=int, help='Percentage of training data to consider in first step', default=100)
        self.parser.add_argument('--mode', type=str, default='rank')
        self.parser.add_argument('--sampler', type=str, default='random-mode')
        self.parser.add_argument('--norm', type=str, default='l1', help=' Normalization (l1(default) or l2)')

        self.parser.add_argument('--subcreate', dest="subcreate", help='Create subgraphs', action='store_true')
        self.parser.add_argument('--subtest', dest = "subtest", help='Test with subgraphs', action='store_true')
        self.parser.add_argument('--minsubsize', type=int, help='Minimum subgraph size', default=50)
        self.parser.add_argument('--topk', type=int, help='Number of top subgraphs to check for evaluation', default=5)
        self.parser.add_argument('--subalgo', type=str, help='Type of subgraph to create', default="avg")
        self.parser.add_argument('--subdistance', type=str, help='Distance function to evaluate subgraphs on', default="avg")
        self.neval = -1
        self.best_valid_score = -1.0
        self.exectimes = []
        self.subgraphs = Subgraphs()
        self.avg_embeddings = []
        self.var_embeddings = []

    def make_subgraphs(self, subType, sorted_triples, mincard, trn_model, sub_algo):
        similar_entities = []
        current = np.zeros(self.args.ncomp, dtype=np.float64)
        count = 0
        prevo = -1
        prevp = -1
        subgraph_logfile="subgraphs-test.log"
        file_data = ""
        for triple in sorted_triples:
            sub = triple[0]
            obj = triple[1]
            rel = triple[2]
            ent = -1
            other_ent = -1
            if subType == SUBTYPE.POS:
                ent = obj
                other_ent = sub
            else:
                #print ("subtype = " , subType)
                ent = sub
                other_ent = obj
            if ent != prevo or rel != prevp:
                if count > mincard:
                    mean = current/count
                    self.avg_embeddings.append(mean)
                    columnsSquareDiff = 0
                    for se in similar_entities:
                        columnsSquareDiff += (trn_model.E[se] - mean) * (trn_model.E[se] - mean)
                    if count > 2:
                        columnsSquareDiff /= (count-1)
                    self.var_embeddings.append(columnsSquareDiff)
                    # add subgraph
                    self.subgraphs.add_subgraphs(subType, prevo, prevp, count, similar_entities)
                    for se in similar_entities:
                        file_data += str(se) + "\n"
                    #print(similar_entities)
                #else:
                #    print("count = ", count , " for ", str(prevo) , " : " , str(prevp))
                count = 0
                prevo = ent
                prevp = rel
                current.fill(0.0)
                similar_entities.clear()
            count += 1
            if sub_algo == "transe":
                current += trn_model.E[other_ent]
            else:
                current += np.dot(trn_model.R[rel], trn_model.E[other_ent])
            similar_entities.append(other_ent)
        # After looping over all triples, add remaining entities to a subgraph
        if count > mincard:
            self.avg_embeddings.append(current/count)
            columnsSquareDiff = 0
            for se in similar_entities:
                columnsSquareDiff += (trn_model.E[se] - mean) * (trn_model.E[se] - mean)
            if count > 2:
                columnsSquareDiff /= (count-1)
            self.var_embeddings.append(columnsSquareDiff)
            # add subgraph
            self.subgraphs.add_subgraphs(subType, prevo, prevp, count, similar_entities)
        print ("# of subgraphs : " , self.subgraphs.get_Nsubgraphs())
        with open(subgraph_logfile, "w") as fout:
            fout.write(file_data)

    def run(self, *args, **kwargs):
        # parse comandline arguments
        self.args = self.parser.parse_args()

        fi = self.args.finfo

        self.file_info = None
        if fi is not None:
            self.file_info = open(fi, "w")


        if self.args.mode == 'rank':
            self.callback = self.ranking_callback
        elif self.args.mode == 'lp':
            self.callback = self.lp_callback
            self.evaluator = LinkPredictionEval
        else:
            raise ValueError('Unknown experiment mode (%s)' % self.args.mode)

        if self.args.subcreate:
            print("UNM: here")
            self.subgraphs_create()
        elif self.args.subtest:
            print("UNM: there")
            self.subgraphs_test()
        else:
            self.train()

    def subgraph_callback(self, trn_model):
        #TODO: use subgraphs to find ranks, scores
        log.info("Computing SUBGRAPH positions and scores for TEST dataset...")
        time_start = timeit.default_timer()
        pos_test = self.ev_test.subgraph_positions(trn_model, self.subgraphs.subgraphs)
        subgraph_ranking_scores(self.fresult, pos_test, 'TEST')
        time_end = timeit.default_timer()
        log.info("Time spent in computing SUBGRAPH positions and scores for TEST dataset = %ds" % (time_end - time_start))
        self.fresult.close()

    def ranking_callback(self, trn, with_eval=False):
        # print basic info
        elapsed = timeit.default_timer() - trn.epoch_start
        self.exectimes.append(elapsed)
        if self.args.no_pairwise:
            log.info("[%3d] time = %ds, loss = %f" % (trn.epoch, elapsed, trn.loss))
        else:
            log.info("[%3d] time = %ds, violations = %d" % (trn.epoch, elapsed, trn.nviolations))
            self.fresult.write("[%3d] time = %ds, violations = %d\n" % (trn.epoch, elapsed, trn.nviolations))

        # if we improved the validation error, store model and calc test error
        if (trn.epoch % self.args.test_all == 0) or with_eval:
            log.info("Computing positions and scores for VALIDATION dataset...")
            time_start = timeit.default_timer()

            plot = False
            if trn.epoch == self.args.me:
                #log.info("PLOT ME\n")
                plot = True

            pos_v, fpos_v = self.ev_valid.positions(trn.model)
            fmrr_valid = ranking_scores(self.fresult, pos_v, fpos_v, trn.epoch, 'VALID')
            time_end = timeit.default_timer()
            log.info("At epoch %d , Time spent in computing positions and scores for VALIDATION dataset = %ds" % (trn.epoch, time_end - time_start))
            self.fresult.write("At epoch %d , Time spent in computing positions and scores for VALIDATION dataset = %ds\n" % (trn.epoch, time_end - time_start))

            log.debug("FMRR valid = %f, best = %f" % (fmrr_valid, self.best_valid_score))
            if fmrr_valid > self.best_valid_score or plot:
                self.best_valid_score = fmrr_valid

                log.info("Computing positions and scores for TEST dataset...")
                time_start = timeit.default_timer()
                pos_t, fpos_t = self.ev_test.positions(trn.model, plot=plot, pagerankMap=self.pagerankMap)
                ranking_scores(self.fresult, pos_t, fpos_t, trn.epoch, 'TEST')
                time_end = timeit.default_timer()
                log.info("At epoch %d, Time spent in computing positions and scores for TEST dataset = %ds" % (trn.epoch, time_end - time_start))
                self.fresult.write("At epoch %d, Time spent in computing positions and scores for TEST dataset = %ds\n" % (trn.epoch, time_end - time_start))

                if self.args.fout is not None:
                    st = {
                        'model': trn.model,
                        'pos test': pos_t,
                        'fpos test': fpos_t,
                        'pos valid': pos_v,
                        'fpos valid': fpos_v,
                        'exectimes': self.exectimes
                    }
                    with open(self.args.fout, 'wb') as fout:
                        pickle.dump(st, fout, protocol=2)
        return True

    def lp_callback(self, m, with_eval=False):
        # print basic info
        elapsed = timeit.default_timer() - m.epoch_start
        self.exectimes.append(elapsed)
        if self.args.no_pairwise:
            log.info("[%3d] time = %ds, loss = %d" % (m.epoch, elapsed, m.loss))
        else:
            log.info("[%3d] time = %ds, violations = %d" % (m.epoch, elapsed, m.nviolations))

        # if we improved the validation error, store model and calc test error
        if (m.epoch % self.args.test_all == 0) or with_eval:
            auc_valid, roc_valid = self.ev_valid.scores(m)

            log.debug("AUC PR valid = %f, best = %f" % (auc_valid, self.best_valid_score))
            if auc_valid > self.best_valid_score:
                self.best_valid_score = auc_valid
                auc_test, roc_test = self.ev_test.scores(m)
                log.debug("AUC PR test = %f, AUC ROC test = %f" % (auc_test, roc_test))

                if self.args.fout is not None:
                    st = {
                        'model': m,
                        'auc pr test': auc_test,
                        'auc pr valid': auc_valid,
                        'auc roc test': roc_test,
                        'auc roc valid': roc_valid,
                        'exectimes': self.exectimes
                    }
                    with open(self.args.fout, 'wb') as fout:
                        pickle.dump(st, fout, protocol=2)
        return True

    def bisect_list_by_percent(self, ll, percentage):
        size = len(ll)
        shuffle(ll)
        first_half_len = (size * percentage) / 100
        second_half_len = size - first_half_len
        first_half = ll[:int(first_half_len)]
        second_half = ll[int(first_half_len):]
        return [first_half, second_half]


    def subgraphs_test(self):
        train_triples, valid_triples, test_triples, sz = self.get_all_triples()
        true_triples = train_triples + test_triples + valid_triples
        if self.args.mode == 'rank':
            self.ev_test = self.evaluator(test_triples, true_triples, self.neval)
            self.ev_valid = self.evaluator(valid_triples,true_triples, self.neval)

        self.subgraphs = Subgraphs.load(self.args.fsub)
        trn_model = Model.load(self.args.fout)
        dataset = self.args.fin.split('/')[-1].split('.')[0]
        algo = self.algo
        epochs = self.args.me
        sub_algo = self.args.subalgo
        mincard = self.args.minsubsize
        outfile = dataset + "-" + algo + "-epochs-" + str(epochs) + "-sub_algo-" + sub_algo + "-tau-"+ str(mincard) +".result"
        fresult = open(outfile, "w")
        self.fresult = fresult
        self.subgraph_callback(trn_model)
    '''
    input:

    output:
            Subgraph embeddings (array of objects of Subgraph class)

    '''
    def subgraphs_create(self):
        train_triples, valid_triples, test_triples, sz = self.get_all_triples()
        xs = train_triples
        print ("Trying to make subgraphs...")
        mincard = self.args.minsubsize
        topk    = self.args.topk
        sub_algo = self.args.subalgo

        dataset = self.args.fin.split('/')[-1].split('.')[0]
        algo = self.algo
        epochs = self.args.me

        print("UNM: reading model")
        results = Model.load(self.args.fout)
        trn_model = results['model']
        print("UNM: reading model : DONE")

        sorted_ps = sorted(xs, key=lambda l : (l[2], l[0]))
        #print ("calling with type = ", SUBTYPE.SPO)
        #print(sorted_ps)
        self.make_subgraphs(SUBTYPE.SPO, sorted_ps, mincard, trn_model, sub_algo)

        sorted_po = sorted(xs, key=lambda l : (l[2], l[1]))
        #print ("calling with type = ", SUBTYPE.POS)
        self.make_subgraphs(SUBTYPE.POS, sorted_po, mincard, trn_model, sub_algo)

        print("UNM: creating subgraphs done")
        trn_model.add_param('S', (self.subgraphs.get_Nsubgraphs(), self.args.ncomp))
        if sub_algo == "avg":
            #print(np.shape(trn.model.S))
            #print(len(trn.model.S))
            trn_model.S = self.avg_embeddings
            #for sube in trn.model.S:
            #    print(type(sube) , " :" , sube)
            #print(type(self.avg_embeddings) , " : " , self.avg_embeddings[self.subgraphs.get_Nsubgraphs()-1])
        elif sub_algo == "var":
            trn_model.S = self.var_embeddings
        #print (triple)
        #print ("dimensions = ", self.args.ncomp)
        #print ("# of triples = ", len(sorted_po))
        subgraph_file_name = dataset + "-" + algo + "-epochs-" + str(epochs) + "-sub_algo-" + sub_algo + "-tau-"+ str(mincard) +".sub"
        self.subgraphs.save(subgraph_file_name)
        # Construct the pickle file name
        model_file_name = dataset + "-" + algo + "-epochs-" + str(epochs) + "-sub_algo-" + sub_algo + "-tau-"+ str(mincard) +".mod"
        trn_model.save(model_file_name)

    def fit_model(self, xs, ys, sz, setup_trainer=True, trainer=None):
        # create sampling objects
        # Sample is given the array of triples.
        # So that it can randomly create other triples that is not part of the original array
        # This is useful to make negative samples
        if self.args.sampler == 'corrupted':
            # create type index, here it is ok to use the whole data
            sampler = sample.CorruptedSampler(self.args.ne, xs, ti)
        elif self.args.sampler == 'random-mode':
            sampler = sample.RandomModeSampler(self.args.ne, [0, 1], xs, sz)
        elif self.args.sampler == 'lcwa':
            sampler = sample.LCWASampler(self.args.ne, [0, 1, 2], xs, sz)
        else:
            raise ValueError('Unknown sampler (%s)' % self.args.sampler)

        if setup_trainer:
            trn = self.setup_trainer(sz, sampler)
        else:
            trn = trainer

        notUpdated = 0
        for count in trn.model.E.updateCounts:
            if count == 0:
                notUpdated += 1
        log.info("Fitting model %s with trainer %s and parameters %s" % (
            trn.model.__class__.__name__,
            trn.__class__.__name__,
            self.args)
        )
        trn.fit(xs, ys)
        # each x in xs is a tuple (SUB, OBJ, PREDicate)
        self.callback(trn, with_eval=True)
        return trn

    def make_graph(self, triples, N, M):
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

    def get_boundaries(self, classes, entity):
        for c in classes:
            if (int(c[2]) <= entity and entity <= int(c[3])):
                return {'left': int(c[2]), 'right':int(c[3])}
        return {'left' : -1, 'right' : -1}
        #raise ValueError("Entity %d should not exist" % (entity))

    def get_all_triples(self):
        # read data
        #with open(self.args.fin, 'rb') as fin:
        #    data = pickle.load(fin)

        file_path             = self.args.fin
        trident_db            = trident.Db(file_path)
        batch_size            = 1000
        percent_valid_triples = 0.01
        percent_test_triples  = 0.01
        if trident_db.n_triples() < 1000:
            batch_size = 100
        batcher = trident.Batcher(file_path, batch_size, 1, percent_valid_triples, percent_test_triples)
        N = trident_db.n_terms()
        M = trident_db.n_relations()

        #N = len(data['entities'])
        #pdb.set_trace()
        #M = len(data['relations'])
        sz = (N, N, M)

        if file_path[-1] != '/':
            file_path = file_path + "/"
        train_triples_series = batcher.load_triples(file_path+"_batch")
        valid_triples_series = batcher.load_triples(file_path+"_batch_valid")
        test_triples_series  = batcher.load_triples(file_path+"_batch_test")

        def parse_triples_series(series):
            s_list = [int(x) for x in series[::3]]
            p_list = [int(x) for x in series[1::3]]
            o_list = [int(x) for x in series[2::3]]
            result = []
            for s,p,o in zip(s_list, p_list, o_list):
                # Note that we are returing SUB,OBJ, PRED
                result.append((s,o,p))
            return result

        train_triples = parse_triples_series(train_triples_series)
        valid_triples = parse_triples_series(valid_triples_series)
        test_triples  = parse_triples_series(test_triples_series)
        return train_triples, valid_triples, test_triples, sz

    def train(self):
        train_triples, valid_triples, test_triples, sz = self.get_all_triples()
        N = sz[0]
        M = sz[2]
        print (type(train_triples))
        print (len(train_triples))
        #true_triples = data['train_subs'] + data['test_subs'] + data['valid_subs']
        #test_triples = data['test_subs']
        true_triples = train_triples + test_triples + valid_triples

        if self.args.mode == 'rank':
            self.ev_test = self.evaluator(test_triples, true_triples, self.neval)
            self.ev_valid = self.evaluator(valid_triples,true_triples, self.neval)
        #elif self.args.mode == 'lp':
        #    self.ev_test = self.evaluator(data['test_subs'], data['test_labels'])
        #    self.ev_valid = self.evaluator(data['valid_subs'], data['valid_labels'])

        # Construct a name for the result file
        # <dataset>-<size of training>-<strategy>-epochs-<number of epochs>-eval-<Evaluate after X epochs>-margin-<margin>.out
        # lubm-full-transe-epochs-500-eval-50-margin-2.0.out
        dataset = self.args.fin.split('/')[-1].split('.')[0]
        size = "full" if self.args.incr == 100 else "exp"
        strategy = self.args.embed
        epochs = self.args.me
        ev = self.args.test_all
        margin = self.args.margin
        outfile = dataset + "-" + size + "-" + strategy + "-epochs-" + str(epochs) + "-eval-" + str(ev) + "-margin-" + str(margin) + ".out"
        fresult = open(outfile, "w")
        self.fresult = fresult

        self.pagerankMap = None
        # If pagerank file is given, then extract the entity-pagerank map
        if self.args.fpagerank is not None:
            with open(self.args.fpagerank, 'r') as fp:
                self.pagerankMap = eval(fp.read())

        global _FILE_TAIL_PREDICTIONS_UNFILTERED
        global _FILE_TAIL_PREDICTIONS_FILTERED
        global _FILE_HEAD_PREDICTIONS_UNFILTERED
        global _FILE_HEAD_PREDICTIONS_FILTERED

        _FILE_TAIL_PREDICTIONS_UNFILTERED = dataset + "-" + _FILE_TAIL_PREDICTIONS_UNFILTERED
        _FILE_TAIL_PREDICTIONS_FILTERED = dataset + "-" + _FILE_TAIL_PREDICTIONS_FILTERED
        _FILE_HEAD_PREDICTIONS_UNFILTERED = dataset + "-" + _FILE_HEAD_PREDICTIONS_UNFILTERED
        _FILE_HEAD_PREDICTIONS_FILTERED = dataset + "-" + _FILE_HEAD_PREDICTIONS_FILTERED

        with open(_FILE_TAIL_PREDICTIONS_UNFILTERED, 'w') as fplot:
            fplot.write("")
        with open(_FILE_TAIL_PREDICTIONS_FILTERED, 'w') as fplot:
            fplot.write("")
        with open(_FILE_HEAD_PREDICTIONS_UNFILTERED, 'w') as fplot:
            fplot.write("")
        with open(_FILE_HEAD_PREDICTIONS_FILTERED, 'w') as fplot:
            fplot.write("")

        # Make a graph from edges in training triples.
        graph_start = timeit.default_timer()
        global graph
        # TODO: for graph use dynamic dict instead of list
        graph = self.make_graph(train_triples, N, M)
        graph_end = timeit.default_timer()
        log.info("Time to build the graph = %ds" %(graph_end - graph_start))
        self.fresult.write("Time to build the graph = %ds\n" %(graph_end - graph_start))

        #sim_start = timeit.default_timer()
        #sim = simrank(graph, N)
        #sim_end = timeit.default_timer()

        #log.info("Time to compute simranks = %ds" %(sim_end - sim_start))
        if self.args.incr != 100:

            # Select 10% of the tuples here

            time_start = timeit.default_timer()
            triples = data['train_subs']
            incremental_batches = self.bisect_list_by_percent(triples, self.args.incr)
            time_end = timeit.default_timer()
            log.info("Time to choose %d%% samples = %ds" % (self.args.incr, time_end-time_start))

            log.info("Total size = %d, %d%% size = %d, %d%% size = %d" % (len(data['train_subs']), self.args.incr, len(incremental_batches[0]), 100-self.args.incr, len(incremental_batches[1])))

            xs = incremental_batches[0]
            ys = np.ones(len(xs))

            time_start = timeit.default_timer()
            trainer = self.fit_model(xs, ys, sz)
            time_end = timeit.default_timer()

            log.info("### Time to fit model for %d%% samples (%d epochs) = %ds" % (self.args.incr, self.args.me, time_end - time_start))
            self.fresult.write("### Time to fit model for %d%% samples (%d epochs) = %ds\n" % (self.args.incr, self.args.me, time_end - time_start))

            log.info("First step finished : ######################")

            time_start = timeit.default_timer()
            countEntities = [0] * N
            for x in xs:
                countEntities[x[0]] += 1
                countEntities[x[1]] += 1

            considered = 0;
            if self.file_info is not None:
                self.file_info.write("Entity (is given) => (embedding of) Entity)\n")

            if self.args.embed is "kognac":
                with open (self.args.ftax, 'r') as ftax:
                    lines = ftax.readlines()

                ranges = [l.split() for l in lines]
                classes = []
                for r in ranges:
                    if (len(r) == 4):
                        classes.append(r)

                classes.sort(key=lambda x:int(x[2]))
            # Apply proper strategy to assign embeddings here
            # If kognac, then read the taxonomy file and based on boundaries of classes, assign embeddings of neighbouring entities.
            # If not, choose other strategy
            # Else, choose random assignment
            lonely = 0
            for entity, count in enumerate(countEntities):
                if count != 0:
                    considered += 1
                else:
                    if self.args.embed is "kognac":
                        # Find the six closest entities that were considered before and take their average

                        boundary = self.get_boundaries(classes, entity)
                        quorum = 6
                        log.info("entity (%d): " % (entity))

                        if (boundary['left'] == -1 and boundary['right'] == -1):
                            # This entitiy is not a part of any class
                            lonely += 1
                            continue

                        neighbours = []
                        if (boundary['left'] == entity):
                            e = entity + 1
                            while(countEntities[e] != 0 and e != boundary['right']-1):
                                neighbours.append(e)
                                if (len(neighbours) == quorum):
                                    break
                                e += 1
                        elif (boundary['right'] == entity):
                            e = entity - 1
                            while (countEntities[e] != 0 and e != boundary['left']):
                                neighbours.append(e)
                                if (len(neighbours) == quorum):
                                    break;
                                e -= 1
                        else:
                            e = entity + 1
                            while(countEntities[e] != 0 and e != boundary['right']-1):
                                neighbours.append(e)
                                if (len(neighbours) == (quorum/2)):
                                    break
                                e += 1

                            required = quorum - (len(neighbours))

                            e = entity - 1
                            while (countEntities[e] != 0 and e != boundary['left']):
                                neighbours.append(e)
                                if (len(neighbours) == required):
                                    break;
                                e -= 1

                        if len(neighbours) > quorum:
                            log.info("More neighbours than the quorum : %d" % (len(neighbours)))

                        quorum = len(neighbours)
                        log.info(" %d neighbours found\n" % (quorum))
                        if quorum != 0:
                            total = np.full((50), 0, dtype=float)
                            for n in neighbours:
                                total += trainer.model.E[n]

                            total /= quorum
                            if self.file_info is not None:
                                for n in neibhours:
                                    self.file_info.write("%d, " % (n))
                                self.file_info.write("\n")

                            trainer.model.E[entity] = total

            time_end = timeit.default_timer()
            log.info("Time spent in assigning new embeddings (Strategy %s) = %ds" % (self.args.embed, time_end - time_start))
            self.fresult.write("Time spent in assigning new embeddings (Strategy %s) = %ds\n" % (self.args.embed, time_end - time_start))

            log.info("!!!!!!!!!!!  %d / %d entities were considered in first batch. !!!!!!!!!!!!!!" % (considered, N))
            log.info("@@@@@@@@  %d entities were lonley (i.e. not a part of any class" % (lonely))

            # Select all tuples
            xs = incremental_batches[0] + incremental_batches[1]
            ys = np.ones(len(xs))

            # Here the trainer is already set-up. So we don't call setup_trainer again.
            # setup_trainer methods initializes the max_epochs parameter which is the number of iterations.
            # We have added a method to the PairwiseStochasticTrainer class which will set the max_epoch for us
            trainer.set_max_epochs(self.args.me/5)
            time_start= timeit.default_timer()
            self.fit_model(xs, ys, sz, setup_trainer=False, trainer=trainer)
            time_end = timeit.default_timer()
            log.info("Time to fit model for 100%% samples (%d epochs) = %ds" % (trainer.max_epochs, time_end - time_start))
            self.fresult.write("Time to fit model for 100%% samples (%d epochs) = %ds\n" % (trainer.max_epochs, time_end - time_start))
        else:
            #print ("UNM$$$ Using 100% data for training... #############")
            xs = train_triples
            ys = np.ones(len(xs))
            time_start= timeit.default_timer()
            trainer = self.fit_model(xs, ys, sz)
            time_end = timeit.default_timer()
            log.info("Time to fit model for 100%% samples (%d epochs) = %ds" % (trainer.max_epochs, time_end - time_start))
            self.fresult.write("Time to fit model for 100%% samples (%d epochs) = %ds\n" % (trainer.max_epochs, time_end - time_start))

            #self.subgraphs_create(xs, ys, sz, trainer)



class FilteredRankingEval(object):

    def __init__(self, xs, true_triples, neval=-1):
        idx = ddict(list)
        tt = ddict(lambda: {'ss': ddict(list), 'os': ddict(list)})
        self.neval = neval
        self.sz = len(xs)
        for s, o, p in xs:
            idx[p].append((s, o))

        for s, o, p in true_triples:
            tt[p]['os'][s].append(o)
            tt[p]['ss'][o].append(s)

        self.idx = dict(idx)
        self.tt = dict(tt)

        self.neval = {}
        for p, sos in self.idx.items():
            if neval == -1:
                self.neval[p] = -1
            else:
                self.neval[p] = np.int(np.ceil(neval * len(sos) / len(xs)))

    def subgraph_positions(self, mdl, subgraphs):
        pos = {}
        # do equivalent of self.prepare_global(mdl)
        count = 0
        sumTailRanks = 0
        sumHeadRanks = 0
        total = 0
        failfile = "failed.log"
        data = ""
        for p, sos in self.idx.items():
            # dictionary with 'tail' as the key, will store positions of H after keeping T and P constant
            ppos = {'head': [], 'tail': []}

            # do self.prepare(mdl, p ) # calcualte ccorr(p , all subgraphs)
            # mdl.S should contain all subgraph embeddings
            SR = ccorr(mdl.R[p], mdl.S)
            for s, o in sos:#[:self.neval[p]]:
                count += 1
                scores_o = np.dot(SR, mdl.E[s]).flatten()
                #print(scores_o)
                #scores_o should contain scores for each subgraph using dot product
                sortidx_o = argsort(scores_o)[::-1]
                # sortidx_o has the indices for sorted subgraph scores
                # Choose topk from this and find out if the answer lies in any of these subgraphs
                found = False
                for rank, index in enumerate(sortidx_o):
                    #print("index = ", index)
                    #print (subgraphs[index].entities)
                    if o in subgraphs[index].entities:
                        found = True
                        break

                sumTailRanks += rank
                if False == found:
                    data += str(o) + "\n"
                    print ("For ", str(s) , ", ", str(p), " subgraph rank(o) = " , rank, " expected o = ", o)
                # rank could be 0 which leads to a possible divide by 0 error
                ppos['tail'].append(rank + 1)

                ocrr = ccorr(mdl.R[p], mdl.E[o])
                scores_s = np.dot(SR, ocrr).flatten()
                #print(scores_s)
                #scores_o should contain scores for each subgraph using dot product
                sortidx_s = argsort(scores_s)[::-1]
                # sortidx_o has the indices for sorted subgraph scores
                # Choose topk from this and find out if the answer lies in any of these subgraphs
                found = False
                for rank, index in enumerate(sortidx_s):
                    if s in subgraphs[index].entities:
                        found = True
                        break

                sumHeadRanks += rank
                total += 1
                if False == found:
                    data += str(s) + "\n"
                    print ("For ", str(o) , ", ", str(p), " subgraph rank(s) = " , rank, " expected s = ", s)
                # rank could be 0 which leads to a possible divide by 0 error
                ppos['head'].append(rank + 1)
            pos[p] = ppos

        print("Mean tail rank = ", sumTailRanks / total)
        print("Mean head rank = ", sumHeadRanks / total)
        with open(failfile, "w") as fout:
            fout.write(data)
        return pos

    def positions(self, mdl, plot=False, pagerankMap=None):
        pos = {}
        fpos = {}

        if hasattr(self, 'prepare_global'):
            self.prepare_global(mdl)

        count = 0
        for p, sos in self.idx.items():
            #pdb.set_trace()

            # There will be just one item in the idx dictionary in case of the un-labelled graph (single-relation graph)
            # So, there will be just one iteration of outer for loop
            # f stands for filtered
            # For unfiltered evaluation, we consider all entities to compute scores with
            # For filtered evaluation, we exclude the neighbours of the entity to compute scores with
            # p might stand for predicate
            # ppos = positions for predicates, where
            # dictionary with 'head' as the key, will store positions of T after keeping H and P constant
            # dictionary with 'tail' as the key, will store positions of H after keeping T and P constant
            ppos = {'head': [], 'tail': []}
            pfpos = {'head': [], 'tail': []}

            # prepare() method adds embeddings of p to embeddings of every entity
            if hasattr(self, 'prepare'):
                #pdb.set_trace()
                # Add the embeddings of p to every entity of this model
                self.prepare(mdl, p)

            #log.info("Prepared\n")
            # For some reason, skip last tuple from all the tuples for relation 'P'
            # neval for every relation is -1
            # self.neval[p] will access the last element and we are skipping the last one by
            # array[:-1]
            #log.info("sos len = %d" % (len(sos)))
            for s, o in sos:#[:self.neval[p]]:
                count += 1
                #print("UNM$$$ ********* score_o = ", self.scores_o(mdl, s, p))
                #print("UNM$$$ ********* score_o = ", np.shape(self.scores_o(mdl, s, p)))
                scores_o = self.scores_o(mdl, s, p).flatten()
                #print("UNM$$$ @@@@@@@@ score_o = ", scores_o)
                #print("UNM$$$ @@@@@@@@ shape(score_o) = ", np.shape(scores_o))
                sortidx_o = argsort(scores_o)[::-1]
                # Sort all the entities (As objects) and find out the index of the "O" in picture
                # Store the index+1 in the ppos['tail]
                rank = np.where(sortidx_o == o)[0][0] + 1
                ppos['tail'].append(rank)

                if plot:
                    inDegree_of_o = num_incoming_neighbours(o, graph)
                    outDegree_of_o = num_outgoing_neighbours(o, graph)
                    totalDegree_of_o = inDegree_of_o + outDegree_of_o
                    inRelations = num_incoming_relations(o, graph)

                    if pagerankMap:
                        with open(_FILE_TAIL_PREDICTIONS_UNFILTERED, 'a') as fplot:
                            fplot.write("%f %d %d %d\n" % (float(pagerankMap[o]) * 100000, 1 if rank <= 10 else 2, totalDegree_of_o, inRelations))
                    else:
                        with open(_FILE_TAIL_PREDICTIONS_UNFILTERED, 'a') as fplot:
                            fplot.write("%d %d %d %d\n" % (inDegree_of_o, 1 if rank <= 10 else 2, totalDegree_of_o, inRelations))
                # In the real data, for relation "P", which entities appear as objects for subject "S"
                rm_idx = self.tt[p]['os'][s]
                # rm_idx is the list of such entities

                # Remove the object "O" that we are currently considering from this list
                rm_idx = [i for i in rm_idx if i != o]

                # Set the scores of KNOWN objects (known truths) to infinity = Filter the entities that already appear as neighbours
                scores_o[rm_idx] = -np.Inf
                sortidx_o = argsort(scores_o)[::-1]
                rank = np.where(sortidx_o == o)[0][0] + 1
                pfpos['tail'].append(rank)

                if plot:
                    if pagerankMap:
                        with open(_FILE_TAIL_PREDICTIONS_FILTERED, 'a') as fplot:
                            fplot.write("%f %d %d %d\n" % (float(pagerankMap[o]) * 100000, 1 if rank <= 10 else 2, totalDegree_of_o, inRelations))
                    else:
                        with open(_FILE_TAIL_PREDICTIONS_FILTERED, 'a') as fplot:
                            fplot.write("%d %d %d %d\n" % (inDegree_of_o, 1 if rank <= 10 else 2, totalDegree_of_o, inRelations))

                ################  HEAD predictions : Keep TAIL/OBJECT constant #######################


                # Unfiltered scores: calculate scores with all entities and sort them
                scores_s = self.scores_s(mdl, o, p).flatten()
                sortidx_s = argsort(scores_s)[::-1]
                rank = np.where(sortidx_s == s)[0][0] + 1
                ppos['head'].append(rank)

                if plot:
                    outDegree_of_s = num_outgoing_neighbours(s, graph)
                    inDegree_of_s= num_incoming_neighbours(s, graph)
                    totalDegree_of_s = outDegree_of_s + inDegree_of_s
                    outRelations = num_outgoing_relations(s, graph)
                    # If pagerank file is provided, write the pagerank of the node instead of the degree
                    if pagerankMap:
                        with open(_FILE_HEAD_PREDICTIONS_UNFILTERED, 'a') as fplot:
                            fplot.write("%f %d %d %d\n" % (float(pagerankMap[s]) * 100000, 1 if rank <= 10 else 2, totalDegree_of_s,  outRelations))
                    else:
                        with open(_FILE_HEAD_PREDICTIONS_UNFILTERED, 'a') as fplot:
                            fplot.write("%d %d %d %d\n" % (outDegree_of_s, 1 if rank <= 10 else 2, totalDegree_of_s, outRelations))

                rm_idx = self.tt[p]['ss'][o]
                rm_idx = [i for i in rm_idx if i != s]
                scores_s[rm_idx] = -np.Inf
                sortidx_s = argsort(scores_s)[::-1]
                rank = np.where(sortidx_s == s)[0][0] + 1
                pfpos['head'].append(rank)
                if plot:
                    # If pagerank file is provided, write the pagerank of the node instead of the degree
                    if pagerankMap:
                        with open(_FILE_HEAD_PREDICTIONS_FILTERED, 'a') as fplot:
                            fplot.write("%f %d %d %d\n" % (float(pagerankMap[s]) * 100000, 1 if rank <= 10 else 2, totalDegree_of_s, outRelations))
                    else:
                        with open(_FILE_HEAD_PREDICTIONS_FILTERED, 'a') as fplot:
                            fplot.write("%d %d %d %d\n" % (outDegree_of_s, 1 if rank <= 10 else 2, totalDegree_of_s,  outRelations))
            pos[p] = ppos
            fpos[p] = pfpos

        if count != self.sz:
            log.info("cnt = %d, self.sz = %d" % (count, self.sz))
        return pos, fpos


class LinkPredictionEval(object):

    def __init__(self, xs, ys):
        ss, os, ps = list(zip(*xs))
        self.ss = list(ss)
        self.ps = list(ps)
        self.os = list(os)
        self.ys = ys

    def scores(self, mdl):
        scores = mdl._scores(self.ss, self.ps, self.os)
        pr, rc, _ = precision_recall_curve(self.ys, scores)
        roc = roc_auc_score(self.ys, scores)
        return auc(rc, pr), roc


def ranking_scores(fresult, pos, fpos, epoch, txt):
    hpos = [p for k in pos.keys() for p in pos[k]['head']]
    tpos = [p for k in pos.keys() for p in pos[k]['tail']]
    fhpos = [p for k in fpos.keys() for p in fpos[k]['head']]
    ftpos = [p for k in fpos.keys() for p in fpos[k]['tail']]
    fmrr = _print_pos(fresult,
        np.array(hpos + tpos),
        np.array(fhpos + ftpos),
        epoch, txt)
    return fmrr

def subgraph_ranking_scores(fresult, pos, txt):
    hpos = [p for k in pos.keys() for p in pos[k]['head']]
    tpos = [p for k in pos.keys() for p in pos[k]['tail']]
    mrr, mean_pos, hits = compute_scores(np.array(hpos+tpos))
    log.info("Subgraph ranking scores : ")
    log.info(
        "%s: MRR = %.2f, Mean Rank = %.2f, Hits@10 = %.2f" %
        (txt, mrr, mean_pos, hits )
    )
    fresult.write(
        "%s: MRR = %.2f, Mean Rank = %.2f, Hits@10 = %.2f\n" %
        (txt, mrr, mean_pos, hits)
    )
    return mrr

def _print_pos(fresult, pos, fpos, epoch, txt):
    mrr, mean_pos, hits = compute_scores(pos)
    fmrr, fmean_pos, fhits = compute_scores(fpos)
    log.info(
        "[%3d] %s: MRR = %.2f/%.2f, Mean Rank = %.2f/%.2f, Hits@10 = %.2f/%.2f" %
        (epoch, txt, mrr, fmrr, mean_pos, fmean_pos, hits, fhits)
    )
    fresult.write(
        "[%3d] %s: MRR = %.2f/%.2f, Mean Rank = %.2f/%.2f, Hits@10 = %.2f/%.2f\n" %
        (epoch, txt, mrr, fmrr, mean_pos, fmean_pos, hits, fhits)
    )
    return fmrr


def compute_scores(pos, hits=10):
    mrr = np.mean(1.0 / pos)
    mean_pos = np.mean(pos)
    hits = np.mean(pos <= hits).sum() * 100
    return mrr, mean_pos, hits


def cardinalities(xs, ys, sz):
    T = to_tensor(xs, ys, sz)
    c_head = []
    c_tail = []
    for Ti in T:
        sh = Ti.tocsr().sum(axis=1)
        st = Ti.tocsc().sum(axis=0)
        c_head.append(sh[np.where(sh)].mean())
        c_tail.append(st[np.where(st)].mean())

    cards = {'1-1': [], '1-N': [], 'M-1': [], 'M-N': []}
    for k in range(sz[2]):
        if c_head[k] < 1.5 and c_tail[k] < 1.5:
            cards['1-1'].append(k)
        elif c_head[k] < 1.5:
            cards['1-N'].append(k)
        elif c_tail[k] < 1.5:
            cards['M-1'].append(k)
        else:
            cards['M-N'].append(k)
    return cards
class Config(object):

    def __init__(self, model, trainer):
        self.model = model
        self.trainer = trainer

    def __getstate__(self):
        return {
            'model': self.model,
            'trainer': self.trainer
        }


class Model(object):
    """
    Base class for all Knowledge Graph models

    Implements basic setup routines for parameters and serialization methods

    Subclasses need to implement:
    - scores(self, ss, ps, os)
    - _gradients(self, xys) for StochasticTrainer
    - _pairwise_gradients(self, pxs, nxs) for PairwiseStochasticTrainer
    C++ : Use virtual functions, make the Model class abstract by having pure virtual functions
    """

    def __init__(self, *args, **kwargs):
        #super(Model, self).__init__(*args, **)
        self.params = {}
        self.hyperparams = {}
        # C++ : No named parameters. Emulate.
        self.add_hyperparam('init', kwargs.pop('init', 'nunif'))

    def add_param(self, param_id, shape, post=None, value=None):
        if value is None:
            value = Parameter(shape, self.init, name=param_id, post=post)
        setattr(self, param_id, value)
        self.params[param_id] = value

    def add_hyperparam(self, param_id, value):
        setattr(self, param_id, value)
        self.hyperparams[param_id] = value

    def __getstate__(self):
        return {
            'hyperparams': self.hyperparams,
            'params': self.params
        }

    def __setstate__(self, st):
        self.params = {}
        self.hyperparams = {}
        for pid, p in st['params'].items():
            self.add_param(pid, None, None, value=p)
        for pid, p in st['hyperparams'].items():
            self.add_hyperparam(pid, p)

    def save(self, fname, protocol=pickle.HIGHEST_PROTOCOL):
        with open(fname, 'wb') as fout:
            pickle.dump(self, fout, protocol=protocol)

    @staticmethod
    def load(fname):
        with open(fname, 'rb') as fin:
            mdl = pickle.load(fin)
        return mdl


class StochasticTrainer(object):
    """
    Stochastic gradient descent trainer with scalar loss function.

    Models need to implement

    _gradients(self, xys)

    to be trained with this class.

    """

    def __init__(self, *args, **kwargs):
        self.model = args[0]
        self.hyperparams = {}
        self.add_hyperparam('max_epochs', kwargs.pop('max_epochs', _DEF_MAX_EPOCHS))
        self.add_hyperparam('nbatches', kwargs.pop('nbatches', _DEF_NBATCHES))
        self.add_hyperparam('learning_rate', kwargs.pop('learning_rate', _DEF_LEARNING_RATE))

        self.post_epoch = kwargs.pop('post_epoch', _DEF_POST_EPOCH)
        self.samplef = kwargs.pop('samplef', _DEF_SAMPLE_FUN)
        pu = kwargs.pop('param_update', AdaGrad)
        self._updaters = {
            key: pu(param, self.learning_rate)
            for key, param in self.model.params.items()
        }

    def set_max_epochs(self, epoch):
        self.max_epochs = epoch

    def __getstate__(self):
        return self.hyperparams

    def __setstate__(self, st):
        for pid, p in st['hyperparams']:
            self.add_hyperparam(pid, p)

    def add_hyperparam(self, param_id, value):
        setattr(self, param_id, value)
        self.hyperparams[param_id] = value

    def fit(self, xs, ys):
        self._optim(list(zip(xs, ys)))

    def _pre_epoch(self):
        self.loss = 0

    def _optim(self, xys):
        # idx = [0,1,2,...., k] where len(xys) = k
        idx = np.arange(len(xys))
        #pdb.set_trace();
        #print("UNM$$$ len(xys) = ", len(xys))
        self.batch_size = len(xys) // self.nbatches
        #print (type(self.batch_size))
        #print(type(xys))
        # A-range (start, stop, jump)
        # For batch size 10 and nbatches 100 and len(xys) = 1000
        # batch_idx = [10,20,30,40,....100,110,....990,1000]
        batch_idx = np.arange(self.batch_size, len(xys), self.batch_size)
        #log.info ("UNM$$$ batch indexes : ", batch_idx)
        #pdb.set_trace()
        for self.epoch in range(1, self.max_epochs + 1):
            # shuffle training examples
            self._pre_epoch()
            shuffle(idx)

            # store epoch for callback
            self.epoch_start = timeit.default_timer()

            # process mini-batches
            # Split the array idx by indexes given in batch_idx
            # batch_idx contains [1414, 2828, 4242, 5656, 7070,...]
            # Thus, batch will contain array of 1414 elements each time
            # entities with ids 0-1413, 1414-2827, 2828-4241 etc.
            #log.info("%d) " % self.epoch)
            for batch in np.split(idx, batch_idx):
                '''
                xys is array of tuple pairs as follows
                ((S1, O1, P1), 1.0 )
                ((S2, O2, P2), 1.0 )
                ((S3, O3, P3), 1.0 )
                ..
                ..
                ((Sn, On, Pn), 1.0 )

                xys[index] will access one of these pairs.
                xys[index][0] will access the triplet.
                xys[index][0][0] will access the subject entity.
                '''
                #log.info("length of minibatch[%d] " % len(batch))
                bxys = [xys[z] for z in batch]
                self._process_batch(bxys)

            # check callback function, if false return
            # post_epoch is the self.callback. It was set in setup_trainer() method
            # of TransEExp
            for f in self.post_epoch:
                if not f(self):
                    break

    def _process_batch(self, xys):
        # if enabled, sample additional examples
        if self.samplef is not None:
            xys += self.samplef(xys)

        if hasattr(self.model, '_prepare_batch_step'):
            self.model._prepare_batch_step(xys)

        # take step for batch
        grads = self.model._gradients(xys)
        self.loss += self.model.loss
        self._batch_step(grads)

    def _batch_step(self, grads):
        for paramID in self._updaters.keys():
            #pdb.set_trace();
            # *grads[paramID] unpacks argument list when calling the function, in this case CTOR
            # Because _updaters is a dictionary
            # _updaters[param] will be the value of type AdaGrad
            # AdaGrad is subclass of ParameterUpdate
            # ParameterUpdate class has a __call__ method
            # This method is called when the instance of ParameterUpdate is called.
            # C++ : functors, overload operator()
            self._updaters[paramID](*grads[paramID])
            #pdb.set_trace();


class PairwiseStochasticTrainer(StochasticTrainer):
    """
    Stochastic gradient descent trainer with pairwise ranking loss functions.

    Models need to implement

    _pairwise_gradients(self, pxs, nxs)

    to be trained with this class.

    """


    def __init__(self, *args, **kwargs):
        super(PairwiseStochasticTrainer, self).__init__(*args, **kwargs)
        self.model.add_hyperparam('margin', kwargs.pop('margin', _DEF_MARGIN))
        fg = kwargs.pop('file_grad', _FILE_GRADIENTS)
        fe = kwargs.pop('file_embed', _FILE_EMBEDDINGS)
        self.file_gradients = None
        self.file_embeddings = None
        self.pickle_file_embeddings = None

        if fg is not None:
            self.file_gradients = open(fg, "w")
        if fe is not None:
            self.file_embeddings = open(fe, "w")
            self.pickle_file_embeddings = open(fe+".pkl", "wb")

    def fit(self, xs, ys):
        # samplef is RandomModeSample set by setup_trainer() method
        if self.samplef is None:
            pidx = np.where(np.array(ys) == 1)[0]
            nidx = np.where(np.array(ys) != 1)[0]
            pxs = [xs[i] for i in pidx]
            self.nxs = [xs[i] for i in nidx]
            self.pxs = int(len(self.nxs) / len(pxs)) * pxs
            xys = list(range(min(len(pxs), len(self.nxs))))
            self._optim(xys)
        else:
            # make a list of tuples such that every entry is the tuple of two tuples (Xs and Ys)
            log.info("Pairwise Stochastic Trainer fit() ")
            self._optim(list(zip(xs, ys)))
            #pdb.set_trace()

            for x in xs:
                # each x is (SUB, OBJ, PREDicate)
                self.model.E.neighbours[x[0]] += 1
                self.model.E.neighbours[x[1]] += 1

            #pdb.set_trace()
            index = 0
            if self.file_gradients is not None:
                self.file_gradients.write("Entity,Degree,#(violations),#(updates)\n")
                for en, ev, ec in zip(self.model.E.neighbours, self.model.E.violations, self.model.E.updateCounts):
                    self.file_gradients.write("%d,%d,%d,%d\n" % (index, en, ev, ec))
                    index += 1

            index = 0
            embeddings_list = []
            for e in self.model.E:
                if self.file_embeddings is not None:
                    embeddings = str(e)
                    embeddings_list.append(e)
                    self.file_embeddings.write("%d,%s\n" % (index, embeddings))
                index += 1
            if self.pickle_file_embeddings is not None:
                pickle.dump(embeddings_list, self.pickle_file_embeddings, protocol=2)

    def _pre_epoch(self):
        self.nviolations = 0
        if self.samplef is None:
            shuffle(self.pxs)
            shuffle(self.nxs)

    def _process_batch(self, xys):
        pxs = []
        nxs = []

        #print ("UNM$$$ process batch of PairWise Stochastic Learner: ", len(xys), " : " , xys[0])
        for xy in xys:

            # samplef is RandomModeSampler
            if self.samplef is not None:
                '''
                Change head or tail of the tuple (H, T, R)
                This code introduces the entities that were not originally present in any tuples
                (in case of T% of tuples processing)
                In other words, it adds 2 negative tuples for every positive tuple
                Why 2 : That's how RandomModeSampler works
                '''
                for nx in self.samplef([xy]):
                    pxs.append(xy)
                    nxs.append(nx)
            else:
                pxs.append((self.pxs[xy], 1))
                nxs.append((self.nxs[xy], 1))

        # take step for batch
        if hasattr(self.model, '_prepare_batch_step'):
            # Not set
            self.model._prepare_batch_step(pxs, nxs)
        #pdb.set_trace()
        #print("UNM$$$ Calling pairwise gradient of HolE : ")
        #print ("UNM$$$ pxs shape = {} , nxs shape = {}".format(np.shape(pxs), np.shape(nxs)))
        grads = self.model._pairwise_gradients(pxs, nxs)

        #pdb.set_trace()
        # update if examples violate margin
        if grads is not None:
            self.nviolations += self.model.nviolations
            #print("UNM$$$ calling _batch_step()")
            self._batch_step(grads)


def sigmoid(fs):
    # compute elementwise gradient for sigmoid
    for i in range(len(fs)):
        if fs[i] > _cutoff:
            fs[i] = 1.0
        elif fs[i] < -_cutoff:
            fs[i] = 0.0
        else:
            fs[i] = 1.0 / (1 + np.exp(-fs[i]))
    return fs[:, np.newaxis]
