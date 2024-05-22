import torch
from decimal import Decimal, getcontext
import numpy as np
import time
#from sklearn.neighbors import KDTree
from scipy.spatial import KDTree
from rich import print


getcontext().prec = 15

def precise_minkowski_distance(u, v, p):
    """
    Precisely calculate the Minkowski distance using the decimal library.

    Parameters:
        u(numpy.ndarray) : The first vector, a numpy ndarray.
        v(numpy.ndarray) : The second vector, a numpy ndarray.
        p(Decimal()) : The order of the Minkowski distance. If p is infinity, Chebyshev distance will be calculated.

    Returns:
    The precisely calculated Minkowski distance.
    """
    if p == Decimal('inf'):
        return max([Decimal(str(abs(ui - vi))) for ui, vi in zip(u, v)])

    distance = Decimal(0)
    for ui, vi in zip(u, v):
        diff = Decimal(ui) - Decimal(vi)
        distance += diff ** Decimal(p)
    

    return distance ** (Decimal(1) / Decimal(p))

def minkowski_distance(data, query_point, p=2):
    if p == float('inf'):
        distances = np.max(np.abs(data - query_point), axis=0)
    else:
        distances = np.sum(np.abs(data - query_point) ** p, axis=0) ** (1 / p)
    return distances


def create_tree(features):
    time1 = time.time()
    leafsize = 40
    compact_nodes = True
    balanced_tree = True
    #kd_tree = KDTree(data=features, leaf_size=leafsize,metric='minkowski',p=float('inf'))
    kd_tree = KDTree(features, leafsize=leafsize)
    time2 = time.time()
    print('[green]K-dimensional Tree is ready, cost[bold blue] {}[/bold blue]s [/green]\n'.format(round(time2-time1,2)))
    return kd_tree
if __name__ =='__main__':

    from rich import print
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([4.0, 5.0, 6.0])
    p = Decimal('inf')
    p=2
    print(Decimal('inf'))

    precise_dist = precise_minkowski_distance(u, v,p)
    print(minkowski_distance(u,v,2))
    print(precise_dist)
    print(float(precise_dist))
    print(str(precise_dist))