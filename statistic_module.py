"""
STATISTIC MODULE


To make operations more convinient np.ndarray should be used
to perform operations.
"""

import numpy as np
import numbers
from numpy.linalg import inv
import matplotlib.pyplot as plt
""" Helper functions """


def to_np(x):
    """
    Makes sure the variable is np compatible\n
    Converts arrays in np arrays\n
    When input is a single value, nothing is done
    """
    if(isinstance(x, np.ndarray)):
        return x
    elif(isinstance(x, list) or
         isinstance(x, tuple)):
        return np.array(x)
    elif(isinstance(x, numbers.Number)):
        return x
    else:
        raise(ValueError)


def update_dict(mydict: dict, **kwargs):
    copy = mydict.copy()
    for kwarg in kwargs:
        if kwarg is not None:
            if kwarg not in copy.keys():
                raise ValueError(f'{kwarg} is not a valid key')
            copy[kwarg] = kwargs[kwarg]
    return copy


def unifEvent(n=None, xstart=0, xend=1):
    if (n is None):
        return np.random.rand()*(xend-xstart)+xstart
    return np.random.rand(n)*(xend-xstart)+xstart


def accRej(n, xstart, xend, ymax, f):
    """
    n: numer of samples reqested
    xstart, xend: x range
    ymax: self explanatory
    """
    x, y = [], []
    while len(x) < n:
        newx = unifEvent(xstart=xstart, xend=xend)
        newy = unifEvent(xend=ymax)

        if (f(newx)-newy >= 0):
            x.append(newx)
            y.append(newy)
            print("accRej: " + "{:.2f}".format(len(x)/n*100), end='\r')
    return x, y


"""
DISTRIBUTION TYPES
"""


def dummy():
    pass


def normDist(x, mean=0, sd=1):
    return np.exp(-0.5*((x-mean)/sd)**2)/(np.sqrt(2*np.pi)*sd)


def expDist(x, mean=1):
    return mean*np.exp(-mean*x)


def unifDist(x, xstart=0, xend=1):
    if (x >= xstart and x <= xend):
        return 1/(xend-xstart)
    return 0


def gen_normDist(mean=0, sd=1):
    return mean + sd*np.sqrt(-2*np.log(np.random.rand()))*np.cos(2*np.pi*np.random.rand())


distRegister = {'normal':       {'func': normDist,
                                 'def': {'mean': 0, 'sd': 1},
                                 'gen': gen_normDist},

                'exp':          {'func': expDist,
                                 'def': {'mean': 1},
                                 'gen': dummy},

                'poisson':      {'fun': dummy,
                                 'def': None,
                                 'gen': dummy},

                'uniform':      {'func': unifDist,
                                 'def': {'xstart': 0, 'xend': 1},
                                 'gen': dummy},

                'binomial':     {'fun': dummy,
                                 'def': None,
                                 'gen': dummy},

                'gamma':        {'fun': dummy,
                                 'def': None,
                                 'gen': dummy},

                'generic':      {'fun': dummy,
                                 'def': None,
                                 'gen': dummy}}


class dice:
    """
    Class to handle random variables.
    The idea is that there are two ways to use this item:
    1) automatically infer the best probability funtion on existing data: arr
    2) given a probability function can generate new events
    """

    def __init__(self, dist) -> None:
        self.dist = dist(dist)

    def roll(self, n):
        if (self.dist.type == 'generic'):
            return accRej(n, *self.dist.xrange, self.dist.ymax, self.dist.f)
        raise (TypeError)


class dist:
    """
    Class to handle distribution funtions
    DOES NOT GENERATE RANDOM NUMBERS

    accepted kwargs:
        - mean
        - sd

    """

    def __init__(self, type='uniform', **kwargs) -> None:
        if type not in distRegister.keys():
            raise TypeError

        self.type = type

        # handle generic type separately
        if type == 'generic':
            f = kwargs['func']
            kwargs.pop('func')
            self.par = kwargs
            self.f = lambda x: f(x, **self.par)
        else:
            self.par = update_dict(distRegister[type]['def'], **kwargs)
            self.f = lambda x: distRegister[type]['func'](x, **self.par)

    def plot(self, ax, start, stop, num, scale=1):
        x = np.linspace(start, stop, num)
        ax.plot(x, self.f(x)*scale)

    def __repr__(self) -> str:
        return "type:{}".format(self.type)


def importData(fname, skip=0):
    return np.loadtxt(fname, skiprows=skip)


def S(i, k, x, y, sig):
    """
    For now y is a
    vector containing error
    y=[y,sigy]
    """
    return np.sum((x**i)*(y**k)/(sig**2))


def linApprox(x, y, ysig):
    D = S(0, 0, x, y, ysig)*S(2, 0, x, y, ysig)-S(1, 0, x, y, ysig)**2
    m = (S(0, 0, x, y, ysig)*S(1, 1, x, y, ysig) -
         S(1, 0, x, y, ysig)*S(0, 1, x, y, ysig))/D
    q = (S(0, 1, x, y, ysig)*S(2, 0, x, y, ysig) -
         S(1, 1, x, y, ysig)*S(1, 0, x, y, ysig))/D
    sdm = np.sqrt(S(0, 0, x, y, ysig)/D)
    sdq = np.sqrt(S(2, 0, x, y, ysig)/D)
    cov = -S(1, 0, x, y, ysig)/D
    return m, q, sdm, sdq, cov


def E(arr):
    """
    Calculates average
    Todo:
        add float check
    """
    return np.sum(arr)/np.size(arr)


def Var(arr, mean):
    """
    Calculates variance
    assumes EQUALLI PROBABLE VALUES
    Todo:
        improve formula to deal with cancellation error
    """
    return np.sum((arr-mean)**2)/(np.size((arr-mean)**2)-1)


def Q2(x, mean, V):
    """
    x: array of random variables\n
    mean: array of means (also single value)\n
    V: correlation matrix\n
    Formula: (x-mean)T*V-1*(x-mean)
    """
    x = to_np(x)
    mean = to_np(mean)
    V = to_np(V)
    return np.transpose(x-mean)@inv(V)@(x-mean)
