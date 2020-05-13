import numpy as np
import pandas as pd


def uniform_dist(n):
    return np.random.rand(n)


def bibeta_dist(n, p1=0.95, a1=2, b1=30, a2=80, b2=2):
    """
    Bi-Beta distribution, calculated as a combination
    of two underlying beta distributions.
    :param n: Size of the output array
    :param p1: probability with which first beta distribution is sampled from
    :param a1: alpha for first beta distribution
    :param b1: beta for first beta distribution
    :param a2: alpha for second beta distribution
    :param b2: beta for second beta distribution
    :return: np array of size N
    """
    use_beta1 = np.random.choice([True, False], size=n, p=[p1, 1 - p1])
    beta1 = np.random.beta(a1, b1, size=use_beta1.sum())
    bibeta = np.random.beta(a2, b2, size=n)
    bibeta[use_beta1] = beta1
    return bibeta


FREQ_DISTS = {
    'uniform': uniform_dist,
    'bibeta': bibeta_dist
}


def edge_gen_from_nx(G, n_days, freq_dist='uniform', **freq_dist_kwargs):
    """
    Based on the static network, generate discrete ("daily") temporal snapshots,
    using a particular frequency distribution to sample edges.
    :param G: static network
    :param n_days: number of days to sample
    :param freq_dist: frequency distribution to use
    :param freq_dist_kwargs: additional freq. dist. parameters
    :return: generator which yields daily edge pd.DataFrames
    """
    edges_static = pd.DataFrame(G.edges(), columns=['a', 'b'])
    N = len(edges_static)

    if freq_dist not in FREQ_DISTS:
        Exception(
            f'{freq_dist} frequency distribution is not supported. '
            f'Available distributions are: {", ".join(FREQ_DISTS.keys())}'
        )

    edges_static['freq'] = FREQ_DISTS[freq_dist](N, **freq_dist_kwargs)

    for day in range(n_days):
        yield day, edges_static[np.random.rand(N) < edges_static.freq]


def visualize_freq_dist(freq_dist, plt_ax=None, **freq_dist_kwargs):
    """
    Sample from the frequency distribution and visualize it.
    :param freq_dist: frequency distribution to visualize
    :param plt_ax: pyplot axes to plot on. If None, global plt is used.
    :param freq_dist_kwargs: additional freq. dist. parameters
    """
    pd.Series(FREQ_DISTS[freq_dist](1000, **freq_dist_kwargs)).hist(bins=100, ax=plt_ax)



cached_ferretti_data = {}

def edge_gen_ferretti(file='all_interaction_10000.csv'):
    global cached_ferretti_data
    if file not in cached_ferretti_data:
        renames = {'ID': 'a', 'ID_2': 'b', 'time': 'day'}
        edges = pd.read_csv(file)[list(renames.keys())].rename(columns=renames)
        edges.day -= edges.day.min()
        edges.sort_values('day', inplace=True)
        cached_ferretti_data[file] = edges

    edges = cached_ferretti_data[file]
    for day in range(edges.day.max() + 1):
        day_edges = edges[edges.day == day]
        # removing b < a edges because model duplicates them later
        yield day, day_edges[day_edges.a < day_edges.b][['a', 'b']]