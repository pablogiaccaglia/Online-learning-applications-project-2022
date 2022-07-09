from Part3.GTS_Learner import GTS_Learner
from Part3.Learner import Learner
import numpy as np
from Knapsack import Knapsack


class CombWrapper:
    """
    Use 1 learner for each campaign then return an optimal super-arm from them
    Can be employed with any learner having this interface:

     ->   def __init__(self, arms, prior_mean, prior_sigma=1)
     ->   def pull_arm(self) -> np.array
            return idx, arms_value
     ->   def update(self, pulled_arm, reward)

    (as GTS_Learner class)
    """
    def __init__(self, learner_constructor, n_campaigns, n_arms, max_budget):  # arms are the budgets (e.g 0,10,20...)
        self.learners = []
        self.max_b = max_budget
        # distribute arms uniformly in range (0, maxbudget)
        self.arms = [int(i * max_budget / n_arms) for i in range(n_arms + 1)]
        # initialize one learner for each campaign
        mean = 350
        var = 90
        for _ in range(n_campaigns):
            self.learners.append(learner_constructor(self.arms, mean, var))

    def pull_super_arm(self) -> np.array:
        """ Return an array budget with the suggested allocation of budgets """
        rewards = []
        for learner in self.learners:
            idx_max, all_arms = learner.pull_arm()
            rewards.append(all_arms)
        #print(f"reward matrix \n{rewards}")
        budgets = np.array(self.arms)
        k = Knapsack(rewards=np.array(rewards), budgets=budgets)
        k.solve()
        arg_max = np.argmax(k.get_output()[0][-1])
        alloc = k.get_output()[1][-1][arg_max]
        super_arms = alloc  # todo the get of result can be optimized

        # return best allocation possible after combinatorial optimization problem
        return super_arms[1:]

    def update_observations(self, super_arm, env_rewards):
        index_arm = self.__indexes_super_arm(super_arm)
        """print(super_arm)
        print(index_arm)
        print(env_rewards)"""
        for i, learner in enumerate(self.learners):
            # update arm of learner if extracted
            if super_arm[i] != 0:
                net_reward = env_rewards[i] - super_arm[i]
                learner.update(index_arm[i], net_reward)

    def get_gp_data(self):
        sigmas = []
        means = []
        for lrn in self.learners:
            sigmas.append(lrn.sigmas)
            means.append(lrn.means)
        return means, sigmas

    def __indexes_super_arm(self, super_arm):
        """Given a super arm return the corresponding index for every learner
            used to understand which arm has been extracted to update its reward """
        indexes = []
        for value_arm in super_arm:
            indexes.append(self.arms.index(value_arm))
        return indexes
