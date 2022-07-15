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
    def __init__(self, learner_constructor, n_campaigns, n_arms, max_budget, arm_distance):  # arms are the budgets (e.g 0,10,20...)
        self.learners = []
        self.max_b = max_budget
        self.last_knapsack_reward = []
        self.arm_distance = arm_distance
        #self.arms = [int(i * max_budget / n_arms) for i in range(0, n_arms + 1)] distribute over max budget
        self.arms = [int(i * arm_distance) for i in range(0, n_arms+1)]  # distribute arms by arm distance

        # this init does not affect GP
        mean = 0
        var = 30
        for _ in range(n_campaigns):
            self.learners.append(learner_constructor(self.arms, mean, var))

    def pull_super_arm(self) -> np.array:
        """ Return an array budget with the super arm allocation of budgets """
        rewards = []
        for learner in self.learners:
            idx_max, all_samples = learner.pull_arm()
            knapsack_r = np.array(all_samples)  # don't remove allocation cost, let learner work with estimated profits
            rewards.append(knapsack_r)

        # add padding for investments up to max budget, needed by knapsack algorithm
        budgets = np.array(self.arms)
        step = self.arm_distance
        start = np.max(budgets) + step
        stop = self.max_b + step
        padding_budgets = np.arange(start, stop, step)
        budgets = np.concatenate([budgets, padding_budgets])

        padding_reward = -1 * np.ones((len(self.learners), len(padding_budgets)))
        """ trial of smooth reduction of dummy values of knapsack padding 
        for i_r, r in enumerate(rewards):
            r_last = r[-1]
            for j, _ in enumerate(padding_reward[i_r]):
                padding_reward[i_r][j] = r_last - step * j"""
        rewards = np.concatenate([np.array(rewards), padding_reward], axis=1)

        # Knapsack execution
        k = Knapsack(rewards=rewards, budgets=budgets)
        k.solve()
        arg_max = np.argmax(k.get_output()[0][-1])
        alloc = k.get_output()[1][-1][arg_max]
        self.last_knapsack_reward = rewards
        # best allocation possible after combinatorial optimization problem
        super_arms = alloc[1:]

        if len(super_arms) > 5:
            # reshape superarms in case of multi campaign knapsack
            # knapsack output [c11,c12,c13,c21,c22,c23] -> [c11,c21, c12, c22. c13.c23]
            group_size = int(len(super_arms) / 5)
            a = np.array(super_arms).reshape((5, group_size))  # reshape in cluster of 3 res x 5 camp
            tmp = np.reshape(a, len(super_arms), order='F')  # reorder per user
            return tmp  # [ctx1|ctx2|ctx3|ctx4]  budget output in context execution

        return super_arms

    def update_observations(self, super_arm, env_rewards, show_warning=False):
        index_arm = self.__indexes_super_arm(super_arm)
        not_pulled = 0
        for i, learner in enumerate(self.learners):
            #if index_arm[i] != 0:   # TRY not update pulling of zero
            reward = np.array(env_rewards).flatten()[i]
            learner.update(index_arm[i], reward)
            if index_arm[i] == 0:
                not_pulled += 1
        if not_pulled > 0 and show_warning:
            print(f"\n\033[93mWarning: {not_pulled}/{len(self.learners)} learners are not pulling arms")

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
