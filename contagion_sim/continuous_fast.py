"""
Improved and adapted versions of ContinuousMultisimModel

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

author: Fabio Mazza
"""
import numpy as np
import pandas as pd
import numba as nb
from continuous_multisim_model import ContinuousMultisimModel

@nb.njit(nb.bool_[:,:](nb.int_[:], nb.bool_[:,:], nb.bool_[:,:]))
def mark_new_inf(dest, S_mat, spread_I):
     ## Run over the edges
        
    new_inf = np.zeros_like(S_mat)
    for i in range(len(dest)):
        recv = dest[i]
        new_inf[recv] = new_inf[recv] | (S_mat[recv] & spread_I[i])
    return new_inf

@nb.njit()
def update_inf(new_inf,dest,S_mat,spread_I):
     ## Run over the edges
    for i in range(len(dest)):
        recv = dest[i]
        new_inf[recv] = new_inf[recv] | (S_mat[recv] & spread_I[i])
#@nb.njit()
"""
@nb.njit(nb.void(nb.bool_[:],nb.bool_[:],nb.bool_[:,:], nb.float64[:,:], nb.float64,nb.int_))
def apply_testing_numba(new_positive,new_negative,I_mat,R_times,mu,today):
    
    new_I = np.zeros_like(I_mat)#.shape,dtype=np.bool)
    new_I[new_positive] = True

    new_I = new_I & ~I_mat
    # Add those to the state
    idcs_I = np.where(new_I)
    #print("\t ",len(idcs_I[0]))
    vals = np.random.geometric(mu, len(idcs_I[0])) + today
    for i in range(len(idcs_I[0])):
        
        I_mat[idcs_I[0][i]][idcs_I[1][i]] = True

        R_times[idcs_I[0][i]][idcs_I[1][i]] = vals[i]


    #new_S = new_negative[:,np.newaxis]
    #I_mat = I_mat & ~new_S
    I_mat[new_negative,:] = False
    R_times[new_negative,:] = np.inf
"""
    

class FasterContinousModel(ContinuousMultisimModel):
    """
    Modified version of ViraTrace ContinuousMultisimModel,
    with higher flexibility, hoping
    for faster simulation times
    """

    def __init__(self, n_nodes, edge_batch_gen, daily_infected,
                 infection_p, recovery_t, recovery_w, n_days, n_sims,
                 daily_tests_positive, daily_tests_random, true_I,
                 contacts_directed=False,
                 strong_negative=False,
                 plt_ax=None, tqdm=None,debug=False):

        ## daily infected is a list of observations

        super().__init__(n_nodes, edge_batch_gen, daily_infected,
                         infection_p,recovery_t, recovery_w, n_days, n_sims,
                         daily_tests_positive, daily_tests_random, true_I,
                         strong_negative, plt_ax, tqdm, debug)
        
        self.contacts_directed = contacts_directed
    
    def update_state(self, spread_I, edges):

        

        #new_inf = np.zeros_like(self.I)

        new_inf = mark_new_inf(edges["a"].to_numpy(), self.S, spread_I)

        self.I = self.I | new_inf
        self.S = self.S & np.logical_not(new_inf)

        self.R_t[np.where(new_inf)] = self.decide_rec_times(new_inf.sum())
    
    def random_binary_array(self, shape, p):

        return np.random.random(shape) < p
    
    def decide_rec_times(self,num):
        return np.random.geometric(1 / self.recovery_t, num) + self.today
        #return np.random.geometric(1 / self.recovery_t, num) + self.today +1

    '''
    def apply_testing(self):
        """
        Apply testing for the day
        """
        #
        # Positives
        
        new_positive = self.daily_positives[self.today]
        new_negative = self.daily_negatives[self.today]

        new_I = new_positive[:,np.newaxis] & np.logical_not(self.I)
        # Add those to the state
        self.I[new_I] = True

        # update R_t per simulation per node
        self.R_t[new_I] = self.decide_rec_times(new_I.sum())

        # DO THE SAME FOR NEGATIVE TESTS
        # remove infection from negative nodes
        #new_S = np.full((self.n_nodes, self.n_sims), False)
        #new_S[new_negative] = True
        new_S = new_negative[:,np.newaxis]
        self.I = self.I & ~new_S
        self.R_t[new_negative] = np.inf
    '''


    def prepare_contacts(self,edges):
        """
        Get the edges in the correct format
        """
        #print("\t\t\t ",self.today)
        if self.contacts_directed:
            return edges
        else:
            edges_inverse = edges.rename(columns={'a': 'b', 'b': 'a'})
            return pd.concat([edges, edges_inverse]).reset_index()

class MultisimRankModel(FasterContinousModel):
    """
    Ranker class for generic problem, with contacts added day after day,
    Fixed infection probability and recovery probability mu
    """
    
    def __init__(self, n_nodes,
                 infection_p, mu, n_days, n_sims,
                 contacts_directed=False,
                 plt_ax=None, tqdm=None,share_init=False):
        
        super().__init__(n_nodes,None, None,
                         infection_p, 1./mu, None, n_days, n_sims,
                         None, None, None, contacts_directed,
                         None, plt_ax, tqdm, False)
            
        self.mu_rec = mu
        self.mat_shape = (n_nodes,n_sims)
        self.positive_tested_set = set()
        self.share_init = share_init
        
        self.all_contacts = []

    def reset_sim(self):

        self.S = np.full(self.mat_shape, True)
        self.I = np.full(self.mat_shape, False)
        self.R_t = np.full(self.mat_shape, np.inf)
        
    def reset_sim_infected(self,p_init):
        """
        Additional method needed to start the simulation 
        when we don't have any observation of infection
        at the beginning of the simulation
        """
        self.R_t = np.full(self.mat_shape, np.inf)
        if self.share_init:
            self.I = np.zeros(self.mat_shape,dtype=np.bool)
            chosen = np.random.rand(self.n_nodes) < p_init
            self.I[chosen,:] = True
            self.R_t[chosen,:] = self.decide_rec_times(chosen.sum())
        else:
            self.I = np.random.random(self.mat_shape) < p_init
            self.R_t[self.I] = self.decide_rec_times(self.I.sum())
        self.S = np.logical_not(self.I)
        
        

    def start_sim(self,p_init=None, day=0 ):
        self.edge_batch_gen = iter(self.all_contacts[day:])
        if day == 0:
            if self.daily_positives[0].sum() < 1:
                if p_init is None:
                    raise ValueError("Parameter p_init missing")
                print("Setting random initial condition")
                self.reset_sim_infected(p_init)
            else:
                self.reset_sim()
        super().run_sim()
    
    def run_sim(self):
        if self.edge_batch_gen is None:
            raise ValueError("Cannot run simulation. Use start_sim method")
        super().run_sim()


    def add_contacts_day(self,day,contacts_df):
        #
        self.all_contacts.append((day, contacts_df))

    def set_max_days(self, n_days):
        self.n_days = n_days

    ### Testing part

    def prepare_testing(self,debug):
        """
        This function is run in the setup of the simulation
        """
        self.daily_positives = [np.zeros(self.n_nodes, dtype=np.bool) for t in range(self.n_days)]
        self.daily_negatives = [np.zeros(self.n_nodes, dtype=np.bool) for t in range(self.n_days)]

    def _propagate_susceptibles(self):
        """
        Apply negative tests backwards in time

        Same code as the last part of ContinuousMultisimModel.prepare_testing()
        """

        self.old_daily_negatives = list(self.daily_negatives)
        negatives = np.full(self.n_nodes, False)
        for i, daily_neg in reversed(list(enumerate(self.daily_negatives))):
            negatives |= daily_neg
            self.daily_negatives[i] = negatives.copy()

    def set_daily_observations(self,day,tested_negative,tested_positive):
        if day >= self.n_days:
            raise ValueError("Cannot put observationsS for days after `n_days` ({})".format(self.n_days))
        self.daily_negatives[day] = tested_negative
        self.daily_positives[day] = tested_positive
        
        index_positive = set(np.where(tested_positive)[0])
        self.positive_tested_set.update(index_positive)

        if day == self.n_days -1:
            print("Propagation")
            self._propagate_susceptibles()

    def set_daily_observations_index(self,day,not_I_index,I_index):
       
        tested_negative = np.zeros(self.n_nodes,dtype=np.bool)
        tested_negative[not_I_index] = True
        tested_positive = np.zeros_like(tested_negative)
        tested_positive[I_index] = True

        self.set_daily_observations(day,tested_negative,tested_positive)

