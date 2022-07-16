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
                 arm_distance=None,
                 is_ucb=False,
                 is_gaussian=False,
                 kwargs=None):  # arms are the budgets (e.g 0,10,20...)

        self.learners = []
        self.max_b = max_budget
        self.last_knapsack_reward = []
        self.is_ucb = is_ucb

        if arm_distance is None:
            arm_distance = max_budget / n_arms

        self.arm_distance = arm_distance
        self.arms = [int(i * arm_distance) for i in range(0, n_arms + 1)]  # distribute arms by arm distance
        self.is_gaussian = is_gaussian

        # this init does not affect GP
        mean = 0
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

        """for r in rewards:
            _min = np.min(r)
            if _min < 0:
                r += _min*-1"""
        padding_reward = 0 * np.ones((len(self.learners), len(padding_budgets)))
        rewards = np.concatenate([np.array(rewards), padding_reward], axis=1)

        k = Knapsack(rewards=np.array(rewards), budgets=budgets)
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
                indexes = self.__indexes_super_arm(super_arm=alloc)  # [0, 3, 5 , 10]
                for learner, indexArm in zip(self.learners, indexes):
                    learner.update_ucbs()
                    cumulative_ucbs[idxAlloc] += learner.ucbs[indexArm]

            arg_max = np.argmax(cumulative_ucbs)
        else:
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

        # return best allocation possible after combinatorial optimization problem
        return super_arms

    def update_observations(self, super_arm, env_rewards, show_warning=False):
        index_arm = self.__indexes_super_arm(super_arm)
        not_pulled = 0
        for i, learner in enumerate(self.learners):
            # if index_arm[i] != 0:   # TRY not update pulling of zero
            reward = np.array(env_rewards).flatten()[i]
            learner.update(index_arm[i], reward)
            if index_arm[i] == 0:
                not_pulled += 1
        if not_pulled > 0 and show_warning:
            print(f"\n\033[93mWarning: {not_pulled}/{len(self.learners)} learners are not pulling arms")

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
