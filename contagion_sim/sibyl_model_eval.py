"""
Improved and adapted versions of MultisimEvaluationModel

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

author: Fabio Mazza
"""
import numpy as np
import numba as nb
from multisim_model_evaluation import MultisimModelEvaluation

@nb.njit(nb.bool_[:, :](nb.int_[:], nb.bool_[:, :], nb.bool_[:, :]))
def mark_new_inf(dest, S_mat, spread_I):
     ## Run over the edges
        
    new_inf = np.zeros_like(S_mat)
    for i in range(len(dest)):
        recv = dest[i]
        new_inf[recv] = new_inf[recv] | (S_mat[recv] & spread_I[i])
    return new_inf

class FasterEvaluationModel(MultisimModelEvaluation):
    """
    Faster version of MultisimModelEvaluation
    with a few quirks:
        tau_backprop_I is the number of days the positive tests (Infected)
                        have to be reapplied backwards in time
    """

    def __init__(self, n_nodes, n_days, n_sims, edge_batch_gen, observations,
                 infection_p, infected_p, recovery_t, recovery_w=None, tau_backprop_I=0,
                 recovery_dist="geometric", directed_edges=False, strong_negative=False,
                 plt_ax=None, tqdm=None):
        super().__init__(n_nodes, n_days, n_sims, edge_batch_gen, observations,
                         infection_p, infected_p, recovery_t, recovery_w, recovery_dist,
                         directed_edges, strong_negative, plt_ax, tqdm)

        self.tau_backprop_I = tau_backprop_I

    def get_daily_positive_obs(self, daily_obs):
        """
        Extract the correct observations for the positives, for today
        Daily obs provided to avoid re-extraction
        """
        obs = self.observations
        if self.tau_backprop_I > 0:
            print(f"Applying tests from {self.today + self.tau_backprop_I} to {self.today}")
            
            sel_mask = (obs.t >= self.today) & (obs.t <= self.today + self.tau_backprop_I) & (obs.state == 1)
            return obs[sel_mask]
        else:
            return daily_obs[daily_obs.state == 1]
    
    
    def _update_state(self, edges, spread_I):
        #new_inf = np.zeros_like(self.I)

        new_inf = mark_new_inf(edges["a"].to_numpy(), self.S, spread_I)

        self.I = self.I | new_inf
        self.S = self.S & np.logical_not(new_inf)

        self.R_t[np.where(new_inf)] = self.recovery_time(new_inf.sum())
        
    def random_binary_array(self, shape, p):

        return np.random.random(shape) < p