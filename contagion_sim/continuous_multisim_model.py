import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, confusion_matrix

from sim_model import AbstractSimModel


class ContinuousMultisimModel(AbstractSimModel):
    """
    Execute continuous multiple contagion spread simulations in parallel
    and analyze the results, including the performance
    of the multi-step risk warnings.
    """

    def __init__(self, n_nodes, edge_batch_gen, daily_infected,
                 infection_p, recovery_t, recovery_w, n_days, n_sims,
                 daily_tests_positive, daily_tests_random,
                 performance_true_I, strong_negative=False,
                 plt_ax=None, tqdm=None):
        """
        :param n_nodes: total number of nodes. Node IDs must go from 0 to n_nodes-1.
        :param edge_batch_gen: generator yielding daily edge pd.DataFrames with
        columns 'a' and 'b' for pairs of nodes
        :param daily_infected: list of infected node vectors per day (ground truth)
        :param infection_p: probability of infected node infecting susceptible
        node during interaction
        :param recovery_t: mean recovery time
        :param recovery_w: recovery time distribution width (std)
        :param n_days: number of days to run the simulation for
        :param daily_tests_positive: number of positive tests daily
        :param daily_tests_random: number of random tests daily
        :param n_sims: total number of simulations to run in parallel
        :param performance_true_I: ground truth infected on the day of analysis
        :param strong_negative: nodes which tested negative at some point
        could not be positive before the test, i.e. nonI = nonR.
        :param plt_ax: pyplot axis to plot the results to
        :param tqdm: tqdm module to used for progress tracking
        """
        super().__init__(plt_ax, tqdm)
        self.n_nodes = n_nodes
        self.edge_batch_gen = edge_batch_gen
        self.daily_infected = daily_infected
        self.infection_p = infection_p
        self.recovery_t = recovery_t
        self.recovery_w = recovery_w
        self.n_days = n_days
        self.n_sims = n_sims
        self.daily_tests_positive = daily_tests_positive
        self.daily_tests_random = daily_tests_random
        self.daily_positives = []
        self.daily_negatives = []
        self.strong_negative = strong_negative
        self.tested_positive = np.full(n_nodes, False)
        self.today = 0
        self.snapshots = []
        self.performance_true_I = performance_true_I

        self.prepare_testing()

        # Model keeps track of S and I states as binary matrices, with R
        # being implicitly defined as ~S & ~I. Additionally, time of recovery
        # (day on which the node recovers) is tracked in the R_t matrix.
        # Each of these three matrices is n_nodes x n_sims big, with each
        # element corresponding to a state of a single node in a single simulation.
        state_dim = (n_nodes, n_sims)

        self.S = np.full(state_dim, True)
        self.I = np.full(state_dim, False)
        self.R_t = np.full(state_dim, np.inf)
        self.tested_positive = np.full(n_nodes, False)


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

    def prepare_testing(self):
        for day, true_I in enumerate(self.daily_infected):
            if true_I is None:
                if self.daily_positives:
                    self.daily_positives.append(self.daily_positives[-1])
                    self.daily_negatives.append(self.daily_negatives[-1])
                else:
                    self.daily_positives.append(np.full(self.n_nodes, False))
                    self.daily_negatives.append(np.full(self.n_nodes, False))
                continue

            # testing counts
            positive_candidates = true_I & ~self.tested_positive
            test_positive_n = min(self.daily_tests_positive, positive_candidates.sum())
            test_random_n = self.daily_tests_random + self.daily_tests_positive - test_positive_n

            # idx of positive tests
            positive_tests_idx = np.random.choice(np.nonzero(positive_candidates)[0],
                                                  size=test_positive_n, replace=False)
            # update global positive tests
            self.tested_positive[positive_tests_idx] = True

            # candidates for random testing are nodes which haven't tested positive yet
            random_candidates = ~self.tested_positive
            # index of random tests
            random_tests_idx = np.random.choice(np.nonzero(random_candidates)[0],
                                                size=test_random_n, replace=False)
            # positive random tests
            random_tests_I = np.full(self.n_nodes, False)
            random_tests_I[random_tests_idx] = true_I[random_tests_idx]

            # update global positive tests
            self.tested_positive |= random_tests_I

            # negative random tests
            random_tests_nonI = np.full(self.n_nodes, False)
            random_tests_nonI[random_tests_idx] = ~true_I[random_tests_idx]

            # new positive nodes from both tests
            new_positive = random_tests_I.copy()
            new_positive[positive_tests_idx] = True

            self.daily_positives.append(new_positive)
            self.daily_negatives.append(random_tests_nonI)

        if self.strong_negative:
            negatives = np.full(self.n_nodes, False)
            for i, daily_neg in reversed(list(enumerate(self.daily_negatives))):
                negatives |= daily_neg
                self.daily_negatives[i] = negatives.copy()

    def apply_testing(self):
        #
        # Positives

        new_positive = self.daily_positives[self.today]

        # new positives per simulation
        new_I = np.full((self.n_nodes, self.n_sims), False)
        new_I[new_positive] = True
        new_I = new_I & ~self.I
        self.I[new_I] = True

        # update R_t per simulation per node
        # self.R_t[new_I] = np.random.normal(self.recovery_t, self.recovery_w, new_I.sum())
        self.R_t[new_I] = np.random.geometric(1 / self.recovery_t, new_I.sum()) + self.today

        #
        # Negatives

        new_negative = self.daily_negatives[self.today]

        # remove infection from negative nodes
        new_S = np.full((self.n_nodes, self.n_sims), False)
        new_S[new_negative] = True
        self.I = self.I & ~new_S
        self.R_t[new_S] = np.inf


    def run_sim(self):
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
                        np.random.geometric(1/self.recovery_t, node_I.sum()) + self.today
                    # self.R_t[node, node_I] = \
                    #     np.random.normal(self.recovery_t, self.recovery_w, node_I.sum())

                # All the infected nodes whose recovery time has passed, are
                # no longer infected.
                self.I[self.R_t <= self.today] = False

                # Take a statistical snapshot
                self.statistical_snapshot()

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
