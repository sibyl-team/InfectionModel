import numpy as np
import pandas as pd
from tqdm import tqdm


class AbstractSimModel:
    """
    Abstract simulation model dealing with the "infrastructure" setup
    """

    def __init__(self, plt_ax=None, tqdm=None):
        self. plt_ax = plt_ax

        if tqdm is None:
            from tqdm import tqdm as tqdm_imp
            self.tqdm = tqdm_imp
        else:
            self.tqdm = tqdm

        self.pbar = None


class SimModel(AbstractSimModel):
    """
    Simple contagion spread simulation model using discrete temporal network.
    Main purpose of this model is to simplify the understanding of the
    MultisimModel which runs multiple simulations in parallel.
    """

    def __init__(self, nodes, edge_batch_gen, infected_p, infection_p,
                 recovery_t, recovery_w, n_days,  plt_ax=None, tqdm=None):
        """
        :param nodes: list of node IDs
        :param edge_batch_gen: generator yielding daily edge pd.DataFrames with
        columns 'a' and 'b' for pairs of nodes
        :param infected_p: probability of each node being infected on day 0
        :param infection_p: probability of infected node infecting susceptible
        node during interaction
        :param recovery_t: mean recovery time
        :param recovery_w: recovery time distribution width (std)
        :param n_days: number of days to run the simulation for
        :param plt_ax: pyplot axis to plot the results to
        :param tqdm: tqdm module to used for progress tracking
        """
        super().__init__(plt_ax, tqdm)
        self.edge_batch_gen = edge_batch_gen
        self.infection_p = infection_p
        self.infected_p = infected_p
        self.recovery_t = recovery_t
        self.recovery_w = recovery_w
        self.n_days = n_days

        self.nodes = pd.DataFrame(index=nodes)
        self.nodes['S'] = True
        self.nodes['I'] = False
        self.nodes['R_t'] = None
        self.nodes['R'] = False

        self.today = 0

        n_infected = int(np.random.binomial(len(nodes), self.infected_p))
        infected_nodes = np.random.choice(nodes, n_infected, replace=False)
        self.infect(infected_nodes)

        self.snapshots = []

    def infect(self, nodes):
        """
        Set nodes to infected.
        :param nodes: nodes to infect
        """
        self.nodes.loc[nodes, 'S'] = False
        self.nodes.loc[nodes, 'I'] = True
        self.nodes.loc[nodes, 'R_t'] = \
            np.random.normal(self.recovery_t, self.recovery_w, len(nodes))

    def recover(self, nodes):
        """
        Set nodes to recovered.
        :param nodes: nodes to recover.
        """
        self.nodes.loc[nodes, 'I'] = False
        self.nodes.loc[nodes, 'R'] = True

    def recovery_tick(self):
        """
        Recovery tick (periodically called) which checks which nodes need to
        be recovered.
        """
        to_recover = self.nodes.index[self.nodes.R_t <= self.today]
        self.recover(to_recover)

    def snapshot(self):
        """
        Create a snapshot of the state-counts for later plotting.
        """
        snap = [self.today] + self.nodes.sum()[['S', 'I', 'R']].to_list()
        self.pbar.set_description('Day:{}\tS:{}\tI:{}\tR:{}'.format(*snap))
        self.snapshots.append(snap)

    def plot_stats(self, log=True):
        """
        Plot the statistical snapshots (S, I and R counts through time).
        :param log: log y axis
        """
        snaps = pd.DataFrame(self.snapshots, columns=['t', 'S', 'I', 'R']).set_index('t')
        snaps.plot(logy=log, ax=self.plt_ax)

    def run_sim(self):
        with tqdm(total=self.n_days) as pbar:
            # progress bar
            self.pbar = pbar
            for day, edges in self.edge_batch_gen:
                # For each day, run the simulation on a daily network.
                # This discrete contagion spread is based on the presumption
                # that no node becomes infective on the same day they were infected.
                pbar.update(1)
                self.today = day

                # infected and susceptible edge-endpoints
                I_a = edges.a.map(self.nodes.I)
                I_b = edges.b.map(self.nodes.I)
                S_a = edges.a.map(self.nodes.S)
                S_b = edges.b.map(self.nodes.S)

                # only I-S edges are considered
                s_nodes = edges.b[I_a & S_b].to_list() + edges.a[S_a & I_b].to_list()

                # choosing the number of edges over which the infection passed
                # by drawing the edge count from a binomial distribution and
                # randomly choosing that many edges
                sample_size = int(np.random.binomial(len(s_nodes), self.infection_p))
                to_infect = np.random.choice(s_nodes, sample_size, replace=False)
                # infecting nodes
                self.infect(to_infect)

                # recovering nodes
                self.recovery_tick()
                # statistics snapshot
                self.snapshot()

                if not self.nodes.I.any():
                    # if no nodes are infected anymore, stop the simulation
                    # changed printf command per debugger output - WMT
                    printf('No more infected nodes, simulation stopped on day {day}.')
                    break
