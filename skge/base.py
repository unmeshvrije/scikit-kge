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

np.random.seed(42)

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
        self.parser.add_argument('--fgrad', type=str, help='Path to store gradient vector updates for each entity', default=None)
        self.parser.add_argument('--fembed', type=str, help='Path to store final embeddings for every entity and relation', default=None)
        self.parser.add_argument('--fin', type=str, help='Path to input data', default=None)
        self.parser.add_argument('--test-all', type=int, help='Evaluate Test set after x epochs', default=10)
        self.parser.add_argument('--no-pairwise', action='store_const', default=False, const=True)
        self.parser.add_argument('--incr', action='store_const', default=False, const=True)
        self.parser.add_argument('--mode', type=str, default='rank')
        self.parser.add_argument('--sampler', type=str, default='random-mode')
        self.parser.add_argument('--norm', type=str, default='l1', help=' Normalization (l1(default) or l2)')
        self.neval = -1
        self.best_valid_score = -1.0
        self.exectimes = []

    def run(self):
        # parse comandline arguments
        self.args = self.parser.parse_args()

        if self.args.mode == 'rank':
            self.callback = self.ranking_callback
        elif self.args.mode == 'lp':
            self.callback = self.lp_callback
            self.evaluator = LinkPredictionEval
        else:
            raise ValueError('Unknown experiment mode (%s)' % self.args.mode)
        self.train()

    def ranking_callback(self, trn, with_eval=False):
        # print basic info
        elapsed = timeit.default_timer() - trn.epoch_start
        self.exectimes.append(elapsed)
        if self.args.no_pairwise:
            log.info("[%3d] time = %ds, loss = %f" % (trn.epoch, elapsed, trn.loss))
        else:
            log.info("[%3d] time = %ds, violations = %d" % (trn.epoch, elapsed, trn.nviolations))

        # if we improved the validation error, store model and calc test error
        if (trn.epoch % self.args.test_all == 0) or with_eval:
            pos_v, fpos_v = self.ev_valid.positions(trn.model)
            fmrr_valid = ranking_scores(pos_v, fpos_v, trn.epoch, 'VALID')

            log.debug("FMRR valid = %f, best = %f" % (fmrr_valid, self.best_valid_score))
            if fmrr_valid > self.best_valid_score:
                self.best_valid_score = fmrr_valid
                pos_t, fpos_t = self.ev_test.positions(trn.model)
                ranking_scores(pos_t, fpos_t, trn.epoch, 'TEST')

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
        first_half = ll[:first_half_len]
        second_half = ll[first_half_len:]
        return [first_half, second_half]

    def train(self):
        # read data
        with open(self.args.fin, 'rb') as fin:
            data = pickle.load(fin)

        N = len(data['entities'])
        #pdb.set_trace()
        M = len(data['relations'])
        sz = (N, N, M)

        true_triples = data['train_subs'] + data['test_subs'] + data['valid_subs']
        if self.args.mode == 'rank':
            self.ev_test = self.evaluator(data['test_subs'], true_triples, self.neval)
            self.ev_valid = self.evaluator(data['valid_subs'], true_triples, self.neval)
        elif self.args.mode == 'lp':
            self.ev_test = self.evaluator(data['test_subs'], data['test_labels'])
            self.ev_valid = self.evaluator(data['valid_subs'], data['valid_labels'])


        if self.args.incr:

            # Select 10% of the tuples here
            triples = data['train_subs']
            incremental_batches = self.bisect_list_by_percent(triples, 10)
            log.info("total size = %d, 10%% size = %d, 90%% size = %d" % (len(data['train_subs']), len(incremental_batches[0]), len(incremental_batches[1])))

            xs = incremental_batches[0]
            ys = np.ones(len(xs))

            # create sampling objects
            if self.args.sampler == 'corrupted':
                # create type index, here it is ok to use the whole data
                sampler = sample.CorruptedSampler(self.args.ne, xs, ti)
            elif self.args.sampler == 'random-mode':
                sampler = sample.RandomModeSampler(self.args.ne, [0, 1], xs, sz)
            elif self.args.sampler == 'lcwa':
                sampler = sample.LCWASampler(self.args.ne, [0, 1, 2], xs, sz)
            else:
                raise ValueError('Unknown sampler (%s)' % self.args.sampler)
            # setup trainer
            trn = self.setup_trainer(sz, sampler)
            log.info("Fitting model %s with trainer %s and parameters %s" % (
                trn.model.__class__.__name__,
                trn.__class__.__name__,
                self.args)
            )
            trn.fit(xs, ys)
            self.callback(trn, with_eval=True)

            # Select all tuples
            log.info("First batch finished : ######################")
            xs = incremental_batches[0] + incremental_batches[1]
            ys = np.ones(len(xs))

            # create sampling objects
            if self.args.sampler == 'corrupted':
                # create type index, here it is ok to use the whole data
                sampler = sample.CorruptedSampler(self.args.ne, xs, ti)
            elif self.args.sampler == 'random-mode':
                sampler = sample.RandomModeSampler(self.args.ne, [0, 1], xs, sz)
            elif self.args.sampler == 'lcwa':
                sampler = sample.LCWASampler(self.args.ne, [0, 1, 2], xs, sz)
            else:
                raise ValueError('Unknown sampler (%s)' % self.args.sampler)
            
            # Don't setup trainer again
            #trn = self.setup_trainer(sz, sampler)
             
            log.info("Fitting model %s with trainer %s and parameters %s" % (
                trn.model.__class__.__name__,
                trn.__class__.__name__,
                self.args)
            )
            trn.fit(xs, ys)
            self.callback(trn, with_eval=True)

            # After this step, we would like to know which entities got better embeddings
            # And then train again
        else:
            xs = data['train_subs']
            ys = np.ones(len(xs))

            # create sampling objects
            if self.args.sampler == 'corrupted':
                # create type index, here it is ok to use the whole data
                sampler = sample.CorruptedSampler(self.args.ne, xs, ti)
            elif self.args.sampler == 'random-mode':
                sampler = sample.RandomModeSampler(self.args.ne, [0, 1], xs, sz)
            elif self.args.sampler == 'lcwa':
                sampler = sample.LCWASampler(self.args.ne, [0, 1, 2], xs, sz)
            else:
                raise ValueError('Unknown sampler (%s)' % self.args.sampler)
            # setup trainer
            trn = self.setup_trainer(sz, sampler)
            log.info("Fitting model %s with trainer %s and parameters %s" % (
                trn.model.__class__.__name__,
                trn.__class__.__name__,
                self.args)
            )
            trn.fit(xs, ys)
            self.callback(trn, with_eval=True)



class FilteredRankingEval(object):

    def __init__(self, xs, true_triples, neval=-1):
        idx = ddict(list)
        tt = ddict(lambda: {'ss': ddict(list), 'os': ddict(list)})
        self.neval = neval
        self.sz = len(xs)
        #pdb.set_trace()
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

    def positions(self, mdl):
        pos = {}
        fpos = {}

        if hasattr(self, 'prepare_global'):
            self.prepare_global(mdl)

        for p, sos in self.idx.items():
            #pdb.set_trace()

            # f stands for filtered (i.e. we will filter the entities that appear in true tuples)
            # p might stand for predicate
            # ppos = positions for predicates, where 
            #'head' will contain the array of most eligible candidates for Head/Subject and 
            #'tail' will contain the array of most eligible candidates for Tail/Objects
            ppos = {'head': [], 'tail': []}
            pfpos = {'head': [], 'tail': []}

            if hasattr(self, 'prepare'):
                #pdb.set_trace()
                self.prepare(mdl, p)

            # For some reason, skip last tuple from all the tuples for relation 'P'
            # neval for every relation is -1
            # self.neval[p] will access the last element and we are skipping the last one by
            # array[:-1]
            for s, o in sos[:self.neval[p]]:
                scores_o = self.scores_o(mdl, s, p).flatten()
                sortidx_o = argsort(scores_o)[::-1]
                # Sort all the entities (As objects) and find out the index of the "O" in picture
                # Store the index+1 in the ppos['tail]
                ppos['tail'].append(np.where(sortidx_o == o)[0][0] + 1)

                #pdb.set_trace()

                # In the real data, for relation "P", which entities appear as objects for subject "S"
                rm_idx = self.tt[p]['os'][s]
                # rm_idx is the list of such entities

                # Remove the object "O" that we are currently considering from this list
                rm_idx = [i for i in rm_idx if i != o]

                # Set the scores of KNOWN objects (known truths) to infinity 
                scores_o[rm_idx] = -np.Inf
                sortidx_o = argsort(scores_o)[::-1]
                pfpos['tail'].append(np.where(sortidx_o == o)[0][0] + 1)

                scores_s = self.scores_s(mdl, o, p).flatten()
                sortidx_s = argsort(scores_s)[::-1]
                ppos['head'].append(np.where(sortidx_s == s)[0][0] + 1)

                rm_idx = self.tt[p]['ss'][o]
                rm_idx = [i for i in rm_idx if i != s]
                scores_s[rm_idx] = -np.Inf
                sortidx_s = argsort(scores_s)[::-1]
                pfpos['head'].append(np.where(sortidx_s == s)[0][0] + 1)
            pos[p] = ppos
            fpos[p] = pfpos

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


def ranking_scores(pos, fpos, epoch, txt):
    hpos = [p for k in pos.keys() for p in pos[k]['head']]
    tpos = [p for k in pos.keys() for p in pos[k]['tail']]
    fhpos = [p for k in fpos.keys() for p in fpos[k]['head']]
    ftpos = [p for k in fpos.keys() for p in fpos[k]['tail']]
    fmrr = _print_pos(
        np.array(hpos + tpos),
        np.array(fhpos + ftpos),
        epoch, txt)
    return fmrr


def _print_pos(pos, fpos, epoch, txt):
    mrr, mean_pos, hits = compute_scores(pos)
    fmrr, fmean_pos, fhits = compute_scores(fpos)
    log.info(
        "[%3d] %s: MRR = %.2f/%.2f, Mean Rank = %.2f/%.2f, Hits@10 = %.2f/%.2f" %
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
        self.batch_size = np.ceil(len(xys) / self.nbatches)
        # A-range (start, stop, jump)
        # For batch size 10 and nbatches 100 and len(xys) = 1000
        # batch_idx = [10,20,30,40,....100,110,....990,1000]
        batch_idx = np.arange(self.batch_size, len(xys), self.batch_size)
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
                bxys = [xys[z] for z in batch]
                self._process_batch(bxys)

            # check callback function, if false return
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
        if fg is not None:
            self.file_gradients = open(fg, "w")
        if fe is not None:
            self.file_embeddings = open(fe, "w") 

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

            log.info("len(Xs) = %6d"% (len(xs)) )
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
            for e in self.model.E:
                if self.file_embeddings is not None:
                    embeddings = str(e)
                    self.file_embeddings.write("%3d,%s\n" % (index, embeddings))
                index += 1

    def _pre_epoch(self):
        self.nviolations = 0
        if self.samplef is None:
            shuffle(self.pxs)
            shuffle(self.nxs)

    def _process_batch(self, xys):
        pxs = []
        nxs = []

        for xy in xys:
            #pdb.set_trace()

            # samplef is RandomModeSampler
            if self.samplef is not None:
                for nx in self.samplef([xy]):
                    pxs.append(xy)
                    nxs.append(nx)
            else:
                #pdb.set_trace()
                pxs.append((self.pxs[xy], 1))
                nxs.append((self.nxs[xy], 1))
                #pdb.set_trace()

        # take step for batch
        if hasattr(self.model, '_prepare_batch_step'):
            self.model._prepare_batch_step(pxs, nxs)
        grads = self.model._pairwise_gradients(pxs, nxs)

        #pdb.set_trace()
        # update if examples violate margin
        if grads is not None:
            self.nviolations += self.model.nviolations
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
