import numpy as np
import pandas as pd
from sim_model import AbstractSimModel


class MultisimModelEvaluation(AbstractSimModel):
    """
    Execute multiple contagion spread simulations in parallel and analyze the
    results, including the performance of the multi-step risk warnings.
    """

    def __init__(self, n_nodes, n_days, n_sims, edge_batch_gen, observations,
                 infection_p, infected_p, recovery_t, recovery_w, recovery_dist='geometric',
                 directed_edges=False, strong_negative=False,
                 plt_ax=None, tqdm=None, debug=False):
        """
        :param n_nodes: total number of nodes. Node IDs must go from 0 to n_nodes-1.
        :param edge_batch_gen: generator yielding daily edge pd.DataFrames with
        columns 'a' and 'b' for pairs of nodes
        :param plt_ax: pyplot axis to plot the results to
        :param tqdm: tqdm module to used for progress tracking
        """
        super().__init__(plt_ax, tqdm)
        self.n_nodes = n_nodes
        self.n_days = n_days
        self.edge_batch_gen = edge_batch_gen
        self.observations = observations
        self.infection_p = infection_p
        self.infected_p = infected_p
        self.recovery_t = recovery_t
        self.recovery_w = recovery_w
        self.recovery_dist = recovery_dist
        self.directed_edges = directed_edges
        self.strong_negative = strong_negative
        self.n_sims = n_sims
        self.today = 0
        self.debug = debug

        # Model keeps track of S and I states as binary matrices, with R
        # being implicitly defined as ~S & ~I. Additionally, time of recovery
        # (day on which the node recovers) is tracked in the R_t matrix.
        # Each of these three matrices is n_nodes x n_sims big, with each
        # element corresponding to a state of a single node in a single simulation.
        state_dim = (n_nodes, n_sims)

        # If each simulation is independent,
        # whole infection matrix is randomly generated.
        self.initial_infected = self.random_binary_array(state_dim, self.infected_p)
        self.I = self.initial_infected.copy()
        n_infected = self.I.sum()

        # Recovery times are randomly drawn from a normal distribution
        self.R_t = np.full(state_dim, np.inf)
        self.R_t[self.I] = self.recovery_time(n_infected)

        # All non-infected nodes are susceptible on day 0, as none are yet recovered
        self.S = ~self.I

        # Indices of initially infected nodes used later
        self.initial_infected_idx = np.argwhere(self.initial_infected).ravel()

    def recovery_time(self, size):
        if self.recovery_dist == 'geometric':
            return np.random.geometric(1/self.recovery_t, size) + self.today
        elif self.recovery_dist == 'normal':
            return np.random.normal(self.recovery_t, self.recovery_w, size) + self.today
        else:
            raise Exception(f'Recovery dist "{self.recovery_dist}" is not supported.')
    
    def get_daily_positive_obs(self, daily_obs):
        """
        Extract the correct observations for the positives, for today
        Daily obs provided to avoid re-extraction
        """
        return daily_obs[daily_obs.state == 1]


    def apply_testing(self):

        daily_observations = self.observations[self.observations.t == self.today]

        #
        # Positives

        new_positive = self.get_daily_positive_obs(daily_observations)

        # new positives per simulation
        new_I = np.full((self.n_nodes, self.n_sims), False)
        new_I[new_positive.a.values] = True
        new_I = new_I & ~self.I
        self.I[new_I] = True

        # update R_t per simulation per node
        # self.R_t[new_I] = np.random.normal(self.recovery_t, self.recovery_w, new_I.sum())
        self.R_t[new_I] = self.recovery_time(new_I.sum())
        #
        # Negatives

        if self.strong_negative:
            previous_observations = self.observations[self.observations.t >= self.today]
            new_negative = previous_observations[previous_observations.state == 0]
        else:
            new_negative = daily_observations[daily_observations.state == 0]

        # remove infection from negative nodes
        new_S = np.full((self.n_nodes, self.n_sims), False)
        new_S[new_negative.a.values] = True
        self.I = self.I & ~new_S
        self.R_t[new_S] = np.inf
        if self.debug:
            print(f"tests - I: {len(new_positive.a.values)}, S: {len(new_negative.a.values)}")
            print(new_positive.a.values, new_negative.a.values)

    def _update_state(self, edges, spread_I):

        # We are not allowed to change the matrix I directly while
        # applying interactions, as multiple-step spreading is not
        # possible in a single day
        new_I = {}
        for a, a_edges in edges.groupby('a'):
            # Grouping together all the input interactions for each
            # 'receiving' node, we set it as infected if it is
            # in fact infected by any of the 'source' nodes
            # and is currently susceptible
            new_I[a] = spread_I[a_edges.index].any(axis=0) & self.S[a]
        for node, node_I in new_I.items():
            # For each updated node (in each simulation), we set it's
            # state to I only if it is either already infected, or if
            # it has been infected today
            self.I[node] = self.I[node] | node_I
            # Each node (in each simulation) is still susceptible only if
            # it has already been susceptible, and hasn't been infected now.
            self.S[node] = ~node_I & self.S[node]
            # Recovery times for newly infected nodes are drawn from a normal distribution
            self.R_t[node, node_I] = self.recovery_time(node_I.sum())

    def run_sim(self):
        # For better understanding of the following process, going through the
        # simple SimModel first is strongly advised.
        with self.tqdm(total=self.n_days) as pbar:
            self.pbar = pbar
            self.pbar.set_description('Starting simulation...')
            for day, edges in self.edge_batch_gen:
                self.today = day
                if day >= self.n_days:
                    return

                self.apply_testing()

                # Interactions are symmetrical, so we duplicate them for the
                # 'other' direction. We think of all interactions as a <- b,
                # meaning b is infecting a.
                if not self.directed_edges:
                    edges_inverse = edges.rename(columns={'a': 'b', 'b': 'a'})
                    edges = pd.concat([edges, edges_inverse], sort=True)
                edges.reset_index(drop=True, inplace=True)

                # Infection vectors of the source ('b') nodes
                spread_I = self.I[edges.b]
                # To be able to reach the target, infection has to pass through
                # a 'filter' with probability 'infection_p'. To simulate this,
                # we build a binary vector passthrough mask with which
                # the original spread vectors are filtered.
                passthrough = self.random_binary_array(spread_I.shape, self.infection_p)
                spread_I = spread_I & passthrough

                self._update_state(edges, spread_I)
                # All the infected nodes whose recovery time has passed, are
                # no longer infected.
                self.I[self.R_t <= self.today] = False

                # Update progress bar
                pbar.update(1)

                if not self.I.any():
                    # If no nodes are infected in none of the simulations, terminate
                    print(f'No more infected nodes in any simulation, terminating on day {day}.')
                    break
