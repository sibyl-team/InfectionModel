import numpy as np
import numba
from multisim_model import MultisimModel

@numba.njit()
def update_inf(new_inf,dest,S_mat,spread_I):
     ## Run over the edges
    for i in range(len(dest)):
        recv = dest[i]
        new_inf[recv] = new_inf[recv] | (S_mat[recv] & spread_I[i])

class MyMultisimModel(MultisimModel):


    def make_new_state(self, spread_I, edges):

        new_inf = np.zeros_like(self.I)
        """
        ## Run over the edges
        for i in range(len(edges)):
            recv = edges.loc[i,"a"]
            #print(f"{edges.loc[i,'b']} -> {recv} :\n\t{spread_I[i].astype(int)}")
            new_inf[recv] = new_inf[recv] | (self.S[recv] & spread_I[i])
            #print(f"\t{inf_trials[recv].astype(int)}")
        """
        update_inf(new_inf, edges["a"].to_numpy(), self.S, spread_I)

        self.I = self.I | new_inf
        self.S = self.S & np.logical_not(new_inf)

        self.R_t[np.where(new_inf)] = \
            np.random.geometric(1 / self.recovery_t, new_inf.sum()) + self.today
        
    def random_binary_array(self, shape, p):
        """
        Create a random binary array.
        :param shape: shape of the output array
        :param p: probability of each element being True
        :return: boolean numpy array
        """
        return np.random.random(shape) < p