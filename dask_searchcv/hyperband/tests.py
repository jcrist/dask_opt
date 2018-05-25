import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
import sklearn
from algs import Hyperband


def accuracy(x, y):
    return (x == y).sum() / len(x)

def classification_data(n=1e3, d=10e3, k=None, p_wrong=0.1):
    if k is None:
        k = d // 100
    n = int(n)
    d = int(d)
    k = int(k)

    X = np.random.randn(n, d)

    w_star = np.zeros(d)
    i = np.random.permutation(d)
    w_star[i[:k]] = np.random.randn(k)

    y = np.sign(X @ w_star)

    flip_idx = np.random.permutation(n)
    y[flip_idx[:int(p_wrong*n)]] *= -1

    X = X.astype(np.float32)
    y = y.astype(np.int64)

    train, test = train_test_split(np.arange(len(y)))
    return X, y, (train, test), w_star

def test_hyperband():
    X, y, (train_, test_), w_star = classification_data()
    classes = np.unique(y)

    model = SGDClassifier(loss='hinge', penalty='elasticnet',
                          alpha=0.0001, l1_ratio=0.5,
                          max_iter=1.0, warm_start=True, average=True)
    model.iters = 0
    # max_iter defines 1 unit of computational resource?
    # whatever, let's use partial_fit for now (which does 1 epoch each time)
    model.loss_fn = sklearn.metrics.hinge_loss
    params = {'alpha': {'type': 'log', 'min': -4, 'max': 0},
              'l1_ratio': {'type': 'linear', 'min': -2, 'max': 0},
              'eta0': {'type': 'log', 'min': -4, 'max': 1},
              'power_t': {'type': 'linear', 'min': 0, 'max': 1},
             }

    model = SGDClassifier(loss='hinge', penalty='elasticnet',
                          learning_rate='invscaling',
                          eta0=1,
                          max_iter=1.0, warm_start=True, average=True)
    alg = Hyperband(params, model, 81, X[train_], y[train_])
    alg.hyperband()
    return alg.history
