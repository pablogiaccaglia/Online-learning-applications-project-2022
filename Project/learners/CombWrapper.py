import numpy as np
from knapsack.Knapsack import Knapsack


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

    def __init__(self,
                 learner_constructor,
                 n_campaigns,
                 n_arms,
                 max_budget,
                 is_ucb = False,
                 is_gaussian = False,
                 kwargs = None):  # arms are the budgets (e.g 0,10,20...)

        self.learners = []
        self.max_b = max_budget
        self.is_ucb = is_ucb
        # distribute arms uniformly in range (0, maxbudget)
        self.arms = [int(i * max_budget / n_arms) for i in range(n_arms + 1)]
        self.is_gaussian = is_gaussian

        # initialize one learner for each campaign
        mean = 350
        var = 90
        for _ in range(n_campaigns):

            if is_gaussian:
                if kwargs is None:
                    self.learners.append(learner_constructor(self.arms, mean, var))
                else:
                    self.learners.append(learner_constructor(self.arms, mean, var, **kwargs))
            else:
                if kwargs is None:
                    self.learners.append(learner_constructor(self.arms))
                else:
                    self.learners.append(learner_constructor(self.arms, **kwargs))

        if len(self.learners) > 0:
            self.bandit_name = self.learners[0].bandit_name

            self.needs_boost = self.learners[0].needs_boost

        else:
            self.bandit_name = 'Bandit'
            self.needs_boost = False

    def pull_super_arm(self) -> np.array:
        """ Return an array budget with the suggested allocation of budgets """
        rewards = []
        for learner in self.learners:
            idx_max, all_arms = learner.pull_arm()
            rewards.append(all_arms)
        # print(f"reward matrix \n{rewards}")
        budgets = np.array(self.arms)
        k = Knapsack(rewards = np.array(rewards), budgets = budgets)
        k.solve()

        if self.is_ucb:
            # update all the upper confidence bounds
            # for learner in self.learners:
            #    learner.update_ucbs()

            # last row allocs
            allocs = k.get_output()[1][-1]

            # compute cumulative ucb for each super arm
            cumulative_ucbs = np.zeros(len(self.arms))

            for idxAlloc, alloc in enumerate(allocs):
                indexes = self.__indexes_super_arm(super_arm = alloc)  # [0, 3, 5 , 10]
                for learner, indexArm in zip(self.learners, indexes):
                    learner.update_ucbs()
                    cumulative_ucbs[idxAlloc] += learner.ucbs[indexArm]

            arg_max = np.argmax(cumulative_ucbs)
        else:
            arg_max = np.argmax(k.get_output()[0][-1])

        alloc = k.get_output()[1][-1][arg_max]
        super_arms = alloc  # todo the get of result can be optimized

        # return best allocation possible after combinatorial optimization problem
        return super_arms[1:]

    def update_observations(self, super_arm, env_rewards):
        index_arm = self.__indexes_super_arm(super_arm)
        """print(gpucb1_super_arm)
        print(index_arm)
        print(env_rewards)"""
        for i, learner in enumerate(self.learners):
            # for each learner update the net reward of the selected arm
            net_reward = env_rewards[i] - super_arm[i]
            learner.update(index_arm[i], net_reward)

    def __indexes_super_arm(self, super_arm):
        """Given a super arm return the corresponding index for every learner
            used to understand which arm has been extracted to update its reward """
        indexes = []
        for value_arm in super_arm:
            indexes.append(self.arms.index(value_arm))
        return indexes

    def reset(self):
        for learner in self.learners:
            learner.reset()

    def get_gp_data(self):
        if self.is_gaussian:
            sigmas = []
            means = []
            for lrn in self.learners:
                sigmas.append(lrn.sigmas)
                means.append(lrn.means)
            return means, sigmas
        else:
            raise Exception("Bandit is not Gaussian")
