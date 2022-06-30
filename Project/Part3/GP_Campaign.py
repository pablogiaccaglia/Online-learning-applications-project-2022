from Project.Campaign import Campaign
import numpy as np

'''
added noise to alphas, ok?
'''
class GP_Campaign(Campaign):
    def __init__(self, id, allocated_budget, alpha_i_max, sigma):
        super().__init__(id, allocated_budget, alpha_i_max)
        self.sigma = sigma

    def change_budget(self, new_budget):
        super().change_budget(new_budget)

    def get_alpha_i_noisy(self, user_alpha_function):
        return min(np.random.normal(user_alpha_function(self.allocated_budget), self.sigma).clip(0.0), self.alpha_i_max)

    def get_alpha_i(self, user_alpha_function):
        super().get_alpha_i(user_alpha_function)

    '''
     Tentative to use the Dirichlet process. A Dirichlet process returns the probabilities of the input 'weights' and
     then it normalizes them to 1. alpha_0 will be at the end and will be the sum of (1-alpha_i)
    '''
    def get_all_alphas_noisy(self, user_alphas):
        alphas = np.array(user_alphas)
        alphas = np.append(alphas, 5-alphas.sum())
        # returns an array with the 5 alphas and the competitor's alpha, all normalized to 1
        return np.random.dirichlet(alphas, 1)


