import numpy as np
from skge.base import Model
from skge.util import grad_sum_matrix, unzip_triples
from skge.param import normalize
import pdb
import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('EX-KG')
class TransE(Model):
    """
    Translational Embeddings of Knowledge Graphs
    """

    def __init__(self, *args, **kwargs):
        super(TransE, self).__init__(*args, **kwargs)
        # call's model class's functions to initialize "Parameter"s
        self.add_hyperparam('sz', args[0])
        self.add_hyperparam('ncomp', args[1])
        self.add_hyperparam('l1', kwargs.pop('l1', True))
        #pdb.set_trace()
        self.add_param('E', (self.sz[0], self.ncomp), post=normalize)
        self.add_param('R', (self.sz[2], self.ncomp))
        log.info("l1 is %r " % (self.l1) )

    def _scores(self, ss, ps, os):
        # This is the dissimilarity measure 'd' from the paper 
        # Head + label and subtract Tail
        # h + l -t OR
        # subject + predicate - object
        # s + p - o
        #pdb.set_trace()
        score = self.E[ss] + self.R[ps] - self.E[os]
        if self.l1:
            #pdb.set_trace()
            score = np.abs(score)
            return -np.sum(score, axis=1)
        else:
            #pdb.set_trace()
            score = score ** 2
            #scoreSubjects = np.sum(self.E[ss] ** 2)
            #scorePredicates = np.sum(self.R[ps] ** 2)
            #scoreObjects = np.sum(self.E[os] ** 2)
            #score = (scoreSubjects ** (1.0/2)) + (scorePredicates ** (1.0/2)) - (scoreObjects ** (1.0/2))
            #return -score
            # don't return negative of np.sum because we are taking square root of this later
            return -np.sum(score, axis=1)

    def _pairwise_gradients(self, pxs, nxs):
        # indices of positive triples
        #pdb.set_trace()
        sp, pp, op = unzip_triples(pxs)
        # indices of negative triples
        sn, pn, on = unzip_triples(nxs)

        # Calculate d(h+l, t) = ||h+l-t||
        #pdb.set_trace();
        pscores = self._scores(sp, pp, op)
        nscores = self._scores(sn, pn, on)
        
        #if not self.l1:
        #    pscores = pscores / 2
        #    nscores = nscores / 2

        # ind contains all violating embeddings
        # all triplets where margin > pscores - nscores
        # i.e. pscores - nscores <= margin
        # So the difference between positive and a negative triple is AT LEAST margin.
        # If it is less than or equal to margin, then that pair is violating the condition
        # In this case we want to move 
        # 1. positive sample's h in direction +X and positive sample's t in -Y
        # 2. negative sample's h in direction -X and negative sample's t in +Y

        ind = np.where(nscores + self.margin > pscores)[0]
        #pdb.set_trace();

        # all examples in batch satify margin criterion
        self.nviolations = len(ind)
        if len(ind) == 0:
            return

        sp = list(sp[ind])
        sn = list(sn[ind])
        pp = list(pp[ind])
        pn = list(pn[ind])
        op = list(op[ind])
        on = list(on[ind])

        #pg = self.E[sp] + self.R[pp] - self.E[op]
        #ng = self.E[sn] + self.R[pn] - self.E[on]
        #pdb.set_trace()
        pg = self.E[op] - self.R[pp] - self.E[sp]
        ng = self.E[on] - self.R[pn] - self.E[sn]
        #pdb.set_trace()

        if self.l1:
            # This part is crucial to understand the derivatives.
            # Because we are doing L1 norm in the score function, Partial derivative of any component (x1) is going to be 1
            # Here pg is the positive gradient, but because we already did +t-h-l (+o-p-s), 
            # we need to inverse the signs of derivatives (i.e. +1 for negative value and -1 for a positive)
            # The sign is nothing but direction we want to move the vector to. 
            # For ng, which is a negative gradient, derivatives correspond to the sign of components, because
            # the negative gradient is supposed to be +t-l-h (+o-p-s)
            pg = np.sign(-pg)
            #ng = -np.sign(-ng)
            ng = np.sign(ng)
        else:
            # Compute L2 norm derivatives which is (h1 + l1 - t1)
            pg = -pg
            ng = ng
            #raise NotImplementedError()

        # entity gradients
        # Sum of sp, op, sn, on = 4 X number of violating tuples
        #pdb.set_trace();
        eidx, Sm, n = grad_sum_matrix(sp + op + sn + on)
        #pdb.set_trace();
        # eidx is the array/list containing all unique entities
        # Sm has number of rows = eidx's length

        # dividing by n is the normalization
        # n contains the list of row sums of matrix Sm
        # This ensures that all values are x such that -1 <= x <=1 
        ge = Sm.dot(np.vstack((pg, -pg, ng, -ng))) / n

        '''
        Sm.shape = 5046 X 10932 
        G = np.vstack(pg,-pg,ng,-ng)
        pg.shape = (10932/4) X 5 (where 5 is number of components in the vector)


        Sm.dot(G) = matrix of shape (5046 X 5) = ge
        Here we have gradients for 5046 entity vectors that will be updated with AdaGrad
        update function

        '''
        #pdb.set_trace();

        # Add the gradient vectors to the list of all gradients for entities and relations
        # This is for instrumentation purpose.
        for e,g in zip(eidx,ge):
            self.E.updateVectors[e].append(g)


        # relation gradients
        ridx, Sm, n = grad_sum_matrix(pp + pn)
        #pdb.set_trace();
        gr = Sm.dot(np.vstack((pg, ng))) / n

        for r,g in zip(ridx, gr):
            self.R.updateVectors[r].append(g)
        #pdb.set_trace();
        return {'E': (ge, eidx), 'R': (gr, ridx)}
