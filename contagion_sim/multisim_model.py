import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, confusion_matrix

from sim_model import AbstractSimModel



class MultisimModel(AbstractSimModel):
    """
    Execute multiple contagion spread simulations in parallel and analyze the
    results, including the performance of the multi-step risk warnings.
    """

    def __init__(self, n_nodes, edge_batch_gen,
                 infected_p, infection_p,
                 recovery_t, recovery_w, n_days, n_sims, noise=None,
                 share_init=False, initial_infected=None,
                 analysis_day=None, analysis_true_I=None,
                 generate_daily_I=False,
                 plt_ax=None, tqdm=None):
        """
        :param n_nodes: total number of nodes. Node IDs must go from 0 to n_nodes-1.
        :param edge_batch_gen: generator yielding daily edge pd.DataFrames with
        columns 'a' and 'b' for pairs of nodes
        :param infected_p: probability of each node being infected on day 0
        :param infection_p: probability of infected node infecting susceptible
        node during interaction
        :param recovery_t: mean recovery time
        :param recovery_w: recovery time distribution width (std)
        :param n_days: number of days to run the simulation for
        :param n_sims: total number of simulations to run in parallel
        :param share_init: if True, initially infected nodes are the same across
        all simulations. Otherwise, each simulation is randomly seeded.
        :param initial_infected: Optional. Binary vector of initially infected
        nodes. Requires share_init to be True.
        :param analysis_day: day on which the performance of multi-step
        risk warning is stored
        :param analysis_true_I: if set, this array is used as a true infected
        vector when analyzing the results.
        :param generate_daily_I: collect daily I from a single simulation.
        :param plt_ax: pyplot axis to plot the results to
        :param tqdm: tqdm module to used for progress tracking
        """
        super().__init__(plt_ax, tqdm)
        self.n_nodes = n_nodes
        self.edge_batch_gen = edge_batch_gen
        self.infection_p = infection_p
        self.infected_p = infected_p
        self.recovery_t = recovery_t
        self.recovery_w = recovery_w
        self.n_days = n_days
        self.n_sims = n_sims
        self.noise = noise
        self.today = 0
        self.snapshots = []
        self.analysis_nb = set()
        self.analysis_day = analysis_day
        self.analysis_true_I = analysis_true_I
        self.analysis = None
        self.generate_daily_I = generate_daily_I
        self.daily_I = []

        # Model keeps track of S and I states as binary matrices, with R
        # being implicitly defined as ~S & ~I. Additionally, time of recovery
        # (day on which the node recovers) is tracked in the R_t matrix.
        # Each of these three matrices is n_nodes x n_sims big, with each
        # element corresponding to a state of a single node in a single simulation.
        state_dim = (n_nodes, n_sims)

        if share_init:
            # If all simulations share the initially infected nodes...
            if initial_infected is not None:
                # ...and if these nodes are provided, we use them...
                single_sim_initial_infected = initial_infected
            else:
                # ...otherwise we create a new vector
                single_sim_initial_infected = self.random_binary_array(n_nodes, self.infected_p)

            n_infected = single_sim_initial_infected.sum()
            self.initial_infected = single_sim_initial_infected
            # initial infection is duplicated across all simulations
            self.I = np.array([single_sim_initial_infected, ] * n_sims).T

            # Default recovery time is inf as susceptible nodes are never recovered
            single_sim_R_t = np.full(n_nodes, np.inf)
            # Each infected node needs a recovery time drawn from a normal distribution
            single_sim_R_t[single_sim_initial_infected] = \
                np.random.normal(self.recovery_t, self.recovery_w, n_infected)
            # These recovery times are again duplicated across all simulations
            self.R_t = np.array([single_sim_R_t, ] * n_sims).T
        else:
            # If each simulation is independent,
            # whole infection matrix is randomly generated.
            self.initial_infected = self.random_binary_array(state_dim, self.infected_p)
            self.I = self.initial_infected.copy()
            n_infected = self.I.sum()

            # Recovery times are randomly drawn from a normal distribution
            self.R_t = np.full(state_dim, np.inf)
            self.R_t[self.I] = np.random.normal(self.recovery_t, self.recovery_w, n_infected)

        # All non-infected nodes are susceptible on day 0, as none are yet recovered
        self.S = ~self.I

        # Indices of initially infected nodes used later
        self.initial_infected_idx = np.argwhere(self.initial_infected).ravel()

    def random_binary_array(self, shape, p):
        """
        Create a random binary array.
        :param shape: shape of the output array
        :param p: probability of each element being True
        :return: boolean numpy array
        """
        return np.random.choice(
            [True, False],
            size=shape,
            p=[p, 1 - p]
        )

    def statistical_snapshot(self):
        """
        Create a snapshot of state statistics - mean percentage of S, I and R
        nodes across all simulations, and update the progress bar description.
        """
        S_p = self.S.mean()
        I_p = self.I.mean()
        R_p = 1 - S_p - I_p
        snap = [self.today, S_p, I_p, R_p]
        self.pbar.set_description(
            f'Day:{self.today + 1}\tS:{int(S_p * 100)}%\tI:{int(I_p * 100)}%\tR:{int(R_p * 100)}%')
        self.snapshots.append(snap)

    def analysis_snapshot(self, edges):
        if self.analysis_day is not None and self.analysis_day <= self.today:
            self.analysis_nb |= set(edges[edges.a.isin(self.initial_infected_idx)].b)
            if self.analysis_day == self.today:
                I = ~self.S
                if self.analysis_true_I is not None:
                    true_I = self.analysis_true_I
                    sim_I = I
                else:
                    true_I = I[:, 0]
                    sim_I = I[:, 1:]
                    self.analysis_true_I = true_I
                sim_score = sim_I.sum(axis=1)
                analysis_stats = pd.DataFrame({'I': true_I, 'score': sim_score},
                                              index=range(self.n_nodes))
                analysis_stats['nb'] = analysis_stats.index.isin(self.analysis_nb)
                analysis_stats['init_I'] = self.initial_infected

                self.analysis = analysis_stats
                
    def make_new_state(self, spread_I, edges):
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
            self.R_t[node, node_I] = \
                np.random.geometric(1 / self.recovery_t, node_I.sum()) + self.today
            # self.R_t[node, node_I] = \
            #     np.random.normal(self.recovery_t, self.recovery_w, node_I.sum())

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
                print(f"Day: {self.today}",end="\r")
                if self.generate_daily_I:
                    self.daily_I.append(self.I[:, 0].copy())

                # Interactions are symmetrical, so we duplicate them for the
                # 'other' direction. We think of all interactions as a <- b,
                # meaning b is infecting a.
                edges_inverse = edges.rename(columns={'a': 'b', 'b': 'a'})
                edges = pd.concat([edges, edges_inverse], sort=True).reset_index()

                # Infection vectors of the source ('b') nodes
                spread_I = self.I[edges.b]
                # To be able to reach the target, infection has to pass through
                # a 'filter' with probability 'infection_p'. To simulate this,
                # we build a binary vector passthrough mask with which
                # the original spread vectors are filtered.
                passthrough = self.random_binary_array(spread_I.shape, self.infection_p)
                spread_I = spread_I & passthrough

                # We are not allowed to change the matrix I directly while
                # applying interactions, as multiple-step spreading is not
                # possible in a single day
                self.make_new_state(spread_I, edges)
                """
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
                    self.R_t[node, node_I] = \
                        np.random.geometric(1 / self.recovery_t, node_I.sum()) + self.today
                    # self.R_t[node, node_I] = \
                    #     np.random.normal(self.recovery_t, self.recovery_w, node_I.sum())
                """
                # All the infected nodes whose recovery time has passed, are
                # no longer infected.
                self.I[self.R_t <= self.today] = False

                # Introduce noise - randomly emerging infection in some percent of population
                if self.noise is not None:
                    emerged_I = self.random_binary_array(self.I.shape, self.noise) & self.S
                    self.I = self.I | emerged_I
                    # Each node (in each simulation) is still susceptible only if
                    # it has already been susceptible, and hasn't been infected now.
                    self.S = ~self.I & self.S
                    # Recovery times for newly infected nodes are drawn from a normal distribution
                    self.R_t[emerged_I] = \
                        np.random.geometric(1 / self.recovery_t, emerged_I.sum()) + self.today
                    # self.R_t[emerged_I] = \
                    #     np.random.normal(self.recovery_t, self.recovery_w, emerged_I.sum()) + self.today


                # Take a statistical snapshot
                self.statistical_snapshot()

                # Take a performance analysis snapshot
                self.analysis_snapshot(edges)

                # Update progress bar
                pbar.update(1)

                if not self.I.any():
                    # If no nodes are infected in none of the simulations, terminate
                    print(f'No more infected nodes in any simulation, terminating on day {day}.')
                    break

    def plot_stats(self, stacked=True):
        """
        Plot statistical snapshots - percentage of S, I and R nodes through time.
        :param stacked: if True, plotting stacked area charts, otherwise line plots.
        """
        snaps = (
            pd.DataFrame(self.snapshots, columns=['t', 'S', 'I', 'R'])
            .set_index('t')[['I', 'S', 'R']]
            .clip(lower=0) * 100
        )
        colors = ['red', 'blue', 'grey']
        if stacked:
            snaps.plot.area(stacked=True, alpha=0.4, color=colors, ylim=(0, 100), ax=self.plt_ax)
        else:
            snaps.plot(alpha=0.4, colors=colors, ax=self.plt_ax)

    def performance_analysis(self, notify_top_p=0.1, display=None):
        """
        Analyze the performance of the multi-step risk warnings.
        :param notify_top_p: top percentage of nodes to set as "predicted infected"
        :param display: display module to use for dataframes. If calling from
        a notebook, pass 'display=display', otherwise leave empty.
        """
        # Quantile is inverse from the 'top P percentage'
        sim_quant = 1 - notify_top_p
        # If no display module, print the tables to stdout
        display = display or print

        # whole analysis table
        an = self.analysis
        # normalize scores
        an.score = an.score / an.score.max()
        # analysis of nodes which weren't initially infected
        noninit = an[~an.init_I]
        # analysis of nodes which weren't initially infected
        # nor in direct contact with them
        nonnb = noninit[~noninit.nb].copy()
        # 'predicted infected' by the simulation
        nonnb['sim'] = nonnb.score >= nonnb.score.quantile(sim_quant)

        # confusion matrix of the direct-contact approach
        conf_mat_nb = confusion_matrix(noninit.I, noninit.nb)
        print('Direct contact performance')
        display(pd.DataFrame(conf_mat_nb,
                             index=['Not infected', 'Infected'],
                             columns=['Not notified', 'Notified']))
        print()

        # confusion matrix of the multi-step approach
        conf_mat_sim = confusion_matrix(nonnb.I, nonnb.sim)
        simtn, simfp, simfn, simtp = conf_mat_sim.ravel()
        simfpr = simfp / (simfp + simtn)
        simtpr = simtp / (simtp + simfn)
        print(f'Simulation performance (for non-direct contact nodes) '
              f'if top {int(notify_top_p * 100)}% nodes are notified')
        display(pd.DataFrame(conf_mat_sim,
                             index=['Not infected', 'Infected'],
                             columns=['Not notified', 'Notified']))

        # ROC curve for the multi-step approach.
        if self.plt_ax is None:
            from matplotlib import pyplot as plt_imp
            plt_ax = plt_imp
        else:
            plt_ax = self.plt_ax
        fpr, tpr, _ = roc_curve(nonnb.I, nonnb.score)
        plt_ax.plot(fpr, tpr, marker='.')
        plt_ax.plot([0, 1], [0, 1], color='black', alpha=0.2)
        plt_ax.scatter([simfpr], [simtpr], color='red', s=100)
        plt_ax.xlabel('False Positive Rate')
        plt_ax.ylabel('True Positive Rate')
        plt_ax.title('ROC curve')
        plt_ax.xlim((0, 1))
        plt_ax.ylim((0, 1))
