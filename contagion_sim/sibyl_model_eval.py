"""
Improved and adapted versions of MultisimEvaluationModel

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

author: Fabio Mazza
"""
import numpy as np
import pandas as pd
import numba as nb
from multisim_model_evaluation import MultisimModelEvaluation

@nb.njit(nb.bool_[:, :] (nb.int_[:], nb.bool_[:, :], nb.bool_[:, :]))
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
    """
    
    def _update_state(self, edges, spread_I):
        #new_inf = np.zeros_like(self.I)

        new_inf = mark_new_inf(edges["a"].to_numpy(), self.S, spread_I)

        self.I = self.I | new_inf
        self.S = self.S & np.logical_not(new_inf)

        self.R_t[np.where(new_inf)] = self.recovery_time(new_inf.sum())