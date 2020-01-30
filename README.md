### How to run

```bash
python skge/run_transe.py --fin data/wn18.bin --test-all 50 --nb 100 --me 500 --margin 2.0 --lr 0.1 --ncomp 50 --fgrad numbers.txt

# Knowledge Graph Embeddings

scikit-kge is a Python library to compute embeddings of knowledge graphs. The
library consists of different building blocks to train and develop models for
knowledge graph embeddings.

To compute a knowledge graph embedding, first instantiate a model and then train it
with desired training method. For instance, to train [holographic embeddings of knowledge graphs](http://arxiv.org/abs/1510.04935) (HolE) with a logistcc loss function:

```python
from skge import HolE, StochasticTrainer

# Load knowledge graph 
# N = number of entities
# M = number of relations
# xs = list of (subject, object, predicte) triples
# ys = list of truth values for triples (1 = true, -1 = false)
N, M, xs, ys = load_data('path to data')

# instantiate HolE with an embedding space of size 100
model = HolE((N, N, M), 100)

# instantiate trainer
trainer = StochasticTrainer(model)

# fit model to knowledge graph
trainer.fit(xs, ys)
```

See the [repository for the experiments in the HolE paper](https://github.com/mnick/holographic-embeddings) for an extensive example how to use this library.

The different available buildings blocks are described in more detail in the following:


### Experiments

* Run transE with maximum epochs 100 and evaluate after 10 iterations.
* Increase margin parameter for experiment (i.e. when splitting training set into two).
* Add a command line parameter for marking bad embeddings after each evaluation on a TEST dataset.

### Model

Instantiating a model, e.g. HolE
```python
model = HolE(
    self.shape,
    self.args.ncomp,
    init=self.args.init,
    rparam=self.args.rparam
)
```

### Trainer

scikit-kge supports two basic ways to train models: 

##### StochasticTrainer (skge.base.StochasticTrainer)
Trains a model with logistic loss function
```python
trainer = StochasticTrainer(
    model,
    nbatches=100,
    max_epochs=500,
    post_epoch=[self.callback],
    learning_rate=0.1
)
self.trainer.fit(xs, ys)
```
##### PairwiseStochasticTrainer (skge.base)
To train a model with pairwise ranking loss
```python
trainer = PairwiseStochasticTrainer(
    model,
    nbatches=100,
    max_epochs=500,
    post_epoch=[self.callback],
    learning_rate=0.1,
    margin=0.2,
    af=af.Sigmoid
)
self.trainer.fit(xs, ys)
```

### Parameter Update
scitkit-kge supports different methods to update the parameters of a model via
the `param_update` keyword of `StochasticTrainer` and `PairwiseStochasticTrainer`.

For instance,
```python
from skge.param import AdaGrad

trainer = StochasticTrainer(
    ...,
    param_update=AdaGrad,
    ...
)
```
uses `AdaGrad` to update the parameter. 

Available parameter update methods are
##### SGD (skge.param.SGD)
Basic stochastic gradient descent. Only parameter is the learning rate.

##### AdaGrad (skge.param.AdaGrad)
AdaGrad method of [Duchi et al., 2011](http://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf). Automatically adapts learning rate based on gradient history. Only parameter is the initial learning rate.

### Sampling
sckit-kge implements different strategies to sample negative examples.


### Using Subgraphs embeddings

Following are the snippets of the shell scripts that create model embeddings, create subgraph embeddings and test them.

First, create/assign various parameters.
```
model = "HolE"
DB = "lubm1" # Name of the dataset
EPOCH = 50 # Number of epochs
SUBTYPE = "avg" # How should the subgraphs be prepared avg or var
MS = 10 # Minimum Subgraph Size
SUBALGO = "hole"
```

Create file names for embeddings, models and subgraphs.
```
model_embeddings_file="/path/to/model/embeddings"
subgraph_embeddings_home="/path/to/result/directory/"
subfile_name=$subgraph_embeddings_home"$DB-$model-epochs-$EPOCH-$SUBTYPE-tau-$MS"".sub"
model_file_name=$subgraph_embeddings_home"$DB-$model-epochs-$EPOCH-$SUBTYPE-tau-$MS"".mod"
```
We use [trident](https://github.com/karmaresearch/trident) to load the datasets.

Create model embeddings (here HolE)
```
python run_hole.py --fin /path/to/trident/db/of/dataset  --test-all 100 --nb 1000 --me $EPOCH --margin 0.2 --lr 0.1 --ncomp 50 --fout $model_embeddings_file
```

Generate subgraphs for $DB with mincard $MS and algo $SUBALGO (Note the `--subcreate` parameter)
```
python run_hole.py --fin /var/scratch/uji300/trident/$DB  --nb 1000 --me $EPOCH --margin 0.2 --lr 0.1 --ncomp 50 --subcreate --minsubsize $MS --subalgo $SUBALGO --subdistance $SUBTYPE --fout $model_embeddings_file
```

Test the accuracy with subgraphs (with the `--subtest` parameter)
```
python run_hole.py --fin /var/scratch/uji300/trident/$DB  --nb 1000 --me $EPOCH --margin 0.2 --lr 0.1 --ncomp 50 --subtest --minsubsize $MS --subalgo $SUBALGO --subdistance $SUBTYPE --fout $model_file_name  --fsub $subfile_name --topk $TOPK
```
