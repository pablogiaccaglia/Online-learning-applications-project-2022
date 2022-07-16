from copy import deepcopy
import numpy as np
from learners.CombWrapper import CombWrapper
from learners.GPTS_Learner import GPTS_Learner
from simulations.Environment import Environment
import matplotlib.pyplot as plt
import progressbar

""" @@@@ simulation SETUP @@@@ """

"""
+---------+---------+--------+
|    .    | Student | Worker |
+---------+---------+--------+
| Family  | usr1    | usr1   |
| Alone   | usr2    | usr3   |
|         |         |        |
+---------+---------+--------+
"""
days = 400
N_user = 400  # reference for what alpha = 1 refers to
reference_price = 3.0
daily_budget = 50 * 5
n_arms = 20
arm_distance = 10
step_k = 5
n_budget = int(daily_budget / step_k)
environment = Environment()

if daily_budget < arm_distance * n_arms:
    raise ValueError("Invalid Configuration for daily budget")

bool_alpha_noise = True
bool_n_noise = True
printBasicDebug = False
printKnapsackInfo = False

# ******* Context initialization ********
breakpoint_1 = 40
breakpoint_2 = 100
context_gen_days = 40
random_init_days = 5
rewards_clairvoyant = []
base_learner_rewards = []
ctx_learner_rewards = []
rewards_knapsack_agg = []
budgets_array = np.array([
    [1, 1, 1, 1, 1],  # u 11
    [1, 1, 1, 1, 1],  # u 12
    [1, 1, 1, 1, 1],  # u 2
    [1, 1, 1, 1, 1]  # u3
]) * 50

context_on = False
target_feature = [True, True]  # start fully aggregated
stop_context = False
context_initialized = False
swap = 1    # set 0 or 1 do decide order of features
target_feature_i = 0

c0_reward = []
ctx_reward = []

ctx_algorithm = GPTS_Learner
# ctx_algorithm = GTS_Learner

img, axss = plt.subplots(nrows=4, ncols=5, figsize=(13, 6))  # alpha plots
axs = axss.flatten()

img2, axss2 = plt.subplots(nrows=2, ncols=2, figsize=(13, 6))  # reward plots
axs2 = axss2.flatten()
""" @@@@ ---------------- @@@@ """


def context_masks(split_family=False, split_student=False):
    # return a mask for the contextual basic blocks
    masks = []
    if not split_family and not split_student:
        return [[1, 1, 1, 1]]

    if split_family and split_student:
        masks.append([1, 0, 0, 0])
        masks.append([0, 1, 0, 0])
        masks.append([0, 0, 1, 0])
        masks.append([0, 0, 0, 1])

        return masks
    if split_family:
        masks.append([1, 1, 0, 0])
        masks.append([0, 0, 1, 1])
    if split_student:
        masks.append([1, 0, 1, 0])
        masks.append([0, 1, 0, 1])
    # [usr1_s, usr1_w, usr2, usr3]
    return masks


def budget_array_from_k_alloc_4(_alloc, flatten=False):
    """ map the super arm result with blocks of budget for every possible context participant """
    # _alloc = [0, 1, 2, 3, 11, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 333]
    if len(_alloc) != 20:
        raise ValueError("Knapsack disaggregated alloc needed")
    alloc_clean = _alloc # 0 already removed
    a = np.array(alloc_clean).reshape((5, -1))  # reshape in cluster of 4 res x 5 camp
    tmp = np.reshape(a, len(alloc_clean), order='F')  # rorder per user
    tmp = tmp.reshape((-1, 5))  # reshape 5 camp x 4 user

    if flatten:
        return tmp.flatten()
    return tmp


def hoeffding_bound(samples, confidence=0.80):
    x = np.mean(samples)
    return x - np.sqrt(-1 * np.log(confidence) / (2 * len(samples)))


""" Istantiate the new learner using a copy of the old trained comb lerner"""
"""def instantiate_new_learner(comb_learner, _contexts, _new_contexts, _ctx_algorithm, _n_campaigns_ctx, _n_arms,
                            _daily_budget):
    old_learners = deepcopy(comb_learner.learners)  # get list of learners
    old_learners = np.array_split(old_learners, len(_contexts))  # split in groups
    new_learners = []
    # n chunks = n new context

    for new_ctx in _new_contexts:  # number of times a chunk need to be allocated
        for j, old_ctx in enumerate(_contexts):  # j will be the index of the chunk to add
            test = np.sum(np.array(new_ctx) * np.array(old_ctx))  # matching condition between contexts
            if test > 0:
                new_learners.append(deepcopy(old_learners[j]))  # add chunk in order
                break
    learner = CombWrapper(ctx_algorithm, _n_campaigns_ctx, _n_arms, _daily_budget, arm_distance,
                is_ucb=False,
                is_gaussian=True)
    learner.learners = np.array(new_learners).flatten()

    return learner"""


def instantiate_new_learner(comb_learner, _contexts, _new_contexts, _ctx_algorithm, _n_campaigns_ctx, _n_arms,
                            _daily_budget):
    learner = CombWrapper(_ctx_algorithm, _n_campaigns_ctx, _n_arms, _daily_budget, arm_distance,
                          is_ucb=False,
                          is_gaussian=True)

    return learner


def reward_plot(active=True):
    if active:
        _axs = axs2  # target axs
        if day % 10 == 0:
            _axs[0].cla()
        if day % 2 == 0:
            x = sim_obj["k_budgets"]
            d = np.linspace(0, len(rewards_clairvoyant), len(rewards_clairvoyant))
            _axs[0].set_xlabel("days")
            _axs[0].set_ylabel("reward")
            _axs[0].plot(d, rewards_clairvoyant)
            _axs[0].plot(d, rewards_clairvoyant)
            _axs[0].plot(d, rewards_knapsack_agg)
            _axs[0].plot(d, base_learner_rewards)
            plt.pause(0.1)


def alpha_plot(comb_learner, active=True):
    if active:
        ordered_rewards = []
        if len(contexts) == 1:
            rewards = sim_obj["rewards_agg"]
        if len(contexts) == 4:
            rewards = sim_obj["rewards_disagg"]
        if len(contexts) == 2:
            rewards = sim_obj["rewards_mix"]

        """for offset in [0, 1, 2, 3]:  # order rewards from groups by users to groups by campaign
            for i_campaign in range(5):
                ordered_rewards.append(rewards[offset + 4 * i_campaign])
        ordered_rewards = np.array_split(ordered_rewards, 4)
        agg_ordered_rewards = []

        for _mask in contexts:  # aggregate knapsack rewards according to actual context
            tmp = np.array(ordered_rewards[0]) * 0
            bool_discount_cost = False
            for i_bit, bit in enumerate(_mask):
                if bit == 1:
                    tmp += ordered_rewards[i_bit]
            agg_ordered_rewards.append(tmp)"""

        _axs = axs  # target axs
        rewards_plot = np.array(rewards).reshape(-1, len(sim_obj["k_budgets"]))
        if day % 20 == 0:
            for i, rw in enumerate(rewards_plot):
                _axs[i].cla()
        if day % 4 == 0:
            x = sim_obj["k_budgets"]
            x2 = comb_learner.arms
            for i, rw in enumerate(rewards_plot):
                # _axs[i].set_xlabel("budget")
                # _axs[i].set_ylabel("profit")
                _axs[i].plot(x, rw)
                # axs[i].plot(x2, comb_learner.last_knapsack_reward[i])
                mean, std = comb_learner.get_gp_data()
                # print(std[0])
                # print(mean[0][0])
                _axs[i].plot(x2, mean[i])
                _axs[i].fill_between(
                    np.array(x2).ravel(),
                    mean[i] - 1.96 * std[i],
                    mean[i] + 1.96 * std[i],
                    alpha=0.5,
                    label=r"95% confidence interval",
                )
            plt.pause(0.3)


# initialize base learner target and shape
contexts = context_masks(split_family=target_feature[0+swap], split_student=target_feature[1-swap])
size_ctx = len(contexts)
start_offset = 0
n_campaigns_ctx = 5 * size_ctx

base_learner = CombWrapper(ctx_algorithm, n_campaigns_ctx, n_arms, daily_budget, arm_distance,
                           is_ucb=False,
                           is_gaussian=True)
super_arm = base_learner.pull_super_arm()
last_superarm = super_arm

boost_start = True
boost_discount = 0.4  # boost discount wr to the highest reward
boost_bias = daily_budget / 20  # ensure a positive reward when all pull 0
# *********************************************************************

for day in progressbar.progressbar(range(days)):
    users, products, campaigns, allocated_budget, prob_users, _ = environment.get_core_entities()
    sim_obj = environment.play_one_day(N_user, reference_price, daily_budget, step_k, bool_alpha_noise,
                                       bool_n_noise, contexts=contexts)  # object with all the day info

    # AGGREGATED
    rewards_knapsack_agg.append(sim_obj["reward_k_agg"])
    alloc, tot = sim_obj["alloc_agg"]
    # print(f"\nAlloc {alloc} tot {tot}")

    # DISAGGREGATED
    rewards_clairvoyant.append(sim_obj["reward_k_disagg"])
    alloc, tot = sim_obj["alloc_disagg"]
    # print(f"Alloc {alloc} tot {tot}")
    # TEST knapsack allocation on env OK
    """alloc_k = budget_array_from_k_alloc_4(alloc, flatten=True)
    sim_obj_2 = environment.replicate_last_day(alloc_k,
                                               N_user,
                                               reference_price,
                                               bool_n_noise,
                                               bool_n_noise,
                                               contexts)

    test_rewards = sim_obj_2["learner_rewards"]
    test = np.sum(test_rewards)"""
    if day == breakpoint_1 or day == breakpoint_2:
        if target_feature_i <= 1:
            context_on = True
            last_superarm = super_arm  # save arm of possible old learner
            target_feature[target_feature_i] = True  # split for next feature
            target_feature_i += 1
            start_day = day
            print(f"context ON - day {day} bool1: {target_feature[0]} bool2: {target_feature[1]}")

    if context_on:
        # generate contexts list of context masks
        new_contexts = context_masks(split_family=target_feature[0+swap], split_student=target_feature[1-swap])
        size_ctx = len(new_contexts)

        if not context_initialized:
            # initialize candidate learner
            n_campaigns_ctx = 5 * size_ctx
            old_learner = base_learner
            ctx_learner = instantiate_new_learner(base_learner,
                                                  contexts,
                                                  new_contexts,
                                                  ctx_algorithm,
                                                  n_campaigns_ctx,
                                                  n_arms,
                                                  daily_budget)

            super_arm = ctx_learner.pull_super_arm()  # pull arm for new learner
            contexts = new_contexts
            context_initialized = True

        # test superarm
        sim_obj_2 = environment.replicate_last_day(super_arm,
                                                   N_user,
                                                   reference_price,
                                                   bool_n_noise,
                                                   bool_n_noise,
                                                   contexts)

        learner_rewards = sim_obj_2["learner_rewards"]
        net_profit_learner = np.sum(learner_rewards)
        ctx_learner.update_observations(super_arm, learner_rewards)
        # collect net profit of learner
        ctx_learner_rewards.append(net_profit_learner)
        base_learner_rewards.append(net_profit_learner)
        ctx_reward.append(net_profit_learner)

        # pull super arm for tomorrow
        super_arm = ctx_learner.pull_super_arm()
        # random init
        if day < breakpoint_1 + random_init_days or breakpoint_2 <= day < breakpoint_2 + random_init_days:
            idx = np.random.choice(len(base_learner.arms) - 1, 5 * len(contexts), replace=True)
            loops = 3 * len(contexts)
            while np.sum(np.array(base_learner.arms)[idx]) >= daily_budget:
                idx = np.random.choice(len(base_learner.arms) - 1 - loops, 5 * len(contexts), replace=True)
                loops += 1
            super_arm = np.array(base_learner.arms)[idx]

        if day == start_day + context_gen_days:
            # stop generator
            print(f"context OFF - day {day}")
            start_offset = start_day + context_gen_days
            context_on = False
            context_initialized = False

            confidence = 0.8
            b1 = hoeffding_bound(c0_reward, confidence=confidence)
            b2 = hoeffding_bound(ctx_reward, confidence=confidence)
            if b1 > b2:
                print(f"The split is NOT worth {b1} > {b2} \tconfidence={confidence}")
                split_condition = False
            else:
                print(f"The split is  worth {b1} < {b2} \tconfidence={confidence}")
                split_condition = True
            ctx_reward = []

            split_condition = True  # Force split to see all splits

            if split_condition:
                base_learner = ctx_learner
            else:
                super_arm = last_superarm

        reward_plot(active=True)
        alpha_plot(ctx_learner)
    else:

        # test superarm
        sim_obj_2 = environment.replicate_last_day(super_arm,
                                                   N_user,
                                                   reference_price,
                                                   bool_n_noise,
                                                   bool_n_noise,
                                                   contexts)

        learner_rewards = sim_obj_2["learner_rewards"]
        net_profit_learner = np.sum(learner_rewards)
        base_learner.update_observations(super_arm, learner_rewards)
        base_learner_rewards.append(net_profit_learner)
        # use as reference the aggregated case
        if day < breakpoint_1:
            c0_reward.append(net_profit_learner)

        # print(f"l {super_arm} {np.sum(learner_rewards)}")

        # pull super arm for tomorrow
        super_arm = base_learner.pull_super_arm()
        # random init
        if day < random_init_days:
            idx = np.random.choice(len(base_learner.arms) - 1, 5 * len(contexts), replace=False)
            while np.sum(np.array(base_learner.arms)[idx]) >= daily_budget:
                idx = np.random.choice(len(base_learner.arms) - 1, 5 * len(contexts), replace=False)
            super_arm = np.array(base_learner.arms)[idx]

        reward_plot(active=True)
        alpha_plot(base_learner)
plt.show()

"""# ********* statistical measures *********************
print(f"super arm:  {super_arm}")
print(f"alloc knap: {alloc[1:]}")

print(f"\n***** FINAL RESULT *****")
print(f"days simulated: {days}")
print(f"total profit:\t {sum(rewards_clairvoyant):.4f}€")
print(f"standard deviation:\t {np.std(rewards_clairvoyant):.4f}€")
print(base_learner_rewards)
print(f"Learner profit:\t {sum(base_learner_rewards):.4f}€")
print("----------------------------")
print(f"average profit:\t {np.mean(rewards_clairvoyant):.4f}€")
print(f"\tstd:\t {np.std(rewards_clairvoyant):.4f}€")
print(f"average reward:\t {np.mean(base_learner_rewards):.4f}€")
print(f"\tstd:\t {np.std(base_learner_rewards):.4f}€")
print(f"average regret\t {np.mean(np.array(rewards_clairvoyant) - np.array(base_learner_rewards)):.4f}€")
print(f"\tstd:\t {np.std(np.array(rewards_clairvoyant) - np.array(base_learner_rewards)):.4f}€")

plt.close()
d = np.linspace(0, len(rewards_clairvoyant), len(rewards_clairvoyant))

img, axss = plt.subplots(nrows=2, ncols=2, figsize=(13, 6))
axs = axss.flatten()

axs[0].set_xlabel("days")
axs[0].set_ylabel("reward")
axs[0].plot(d, rewards_clairvoyant)
axs[0].plot(d, base_learner_rewards)

axs[1].set_xlabel("days")
axs[1].set_ylabel("cumulative reward")
axs[1].plot(d, np.cumsum(rewards_clairvoyant))
axs[1].plot(d, np.cumsum(base_learner_rewards))

axs[2].set_xlabel("days")
axs[2].set_ylabel("cumulative regret")
axs[2].plot(d, np.cumsum(np.array(rewards_clairvoyant) - np.array(base_learner_rewards)))

axs[3].set_xlabel("days")
axs[3].set_ylabel("regret")
axs[3].plot(d, np.array(rewards_clairvoyant) - np.array(base_learner_rewards))
plt.show()
"""
