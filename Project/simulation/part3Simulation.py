import numpy as np
import pandas as pd
import progressbar

from Knapsack import Knapsack
from Part3.GTS_Learner import GTS_Learner
from Part3.GPTS_Learner import GPTS_Learner
from Part3.CombWrapper import CombWrapper
from Part3.GPTS_Learner import GPTS_Learner
from simulation.Environment import Environment
import matplotlib.pyplot as plt

# this simulation can be used for both part 2 and 3

""" @@@@ simulation SETUP @@@@ """
days = 100
N_user = 400  # reference for what alpha = 1 refers to
reference_price = 4.0
daily_budget = 70 * 5
n_arms = 20
step_k = 2
environment = Environment()

boost_start = True
boost_discount = 0.5  # boost discount wr to the highest reward
boost_bias = daily_budget / 5 # ensure a positive reward when all pull 0
bool_alpha_noise = False
bool_n_noise = False
printBasicDebug = False
printKnapsackInfo = True
runAggregated = False  # mutual exclusive with run disaggregated
""" Change here the wrapper for the core bandit algorithm """
# comb_learner = CombWrapper(GTS_Learner, 5, n_arms, daily_budget)
comb_learner = CombWrapper(GPTS_Learner, 5, n_arms, daily_budget)
""" @@@@ ---------------- @@@@ """


def table_metadata(n_prod, n_users, avail_budget):
    _col_labels = [str(budget) for budget in avail_budget]

    _row_label_rewards = []
    _row_labels_dp_table = ['0']
    for i in range(1, n_prod + 1):
        for j in range(1, n_users + 1):
            # Cij -> campaign i and user j
            _row_label_rewards.append("C" + str(i) + str(j))
            _row_labels_dp_table.append("+C" + str(i) + str(j))
    return _row_label_rewards, _row_labels_dp_table, _col_labels


def set_budgets_knapsack_env(knapsack_alloc):
    # add knapsack offset
    for i, b in enumerate(knapsack_alloc[1:]):
        environment.set_campaign_budget(i, b)


def set_budgets_arm_env(s_arm):
    for i, b in enumerate(s_arm):
        environment.set_campaign_budget(i, b)

"""
We use knapsack aggregated for benchmark since running knapsack disaggregated is too computational demanding"""

rewards_knapsack_agg = []
rewards_disaggregated = []

learner_rewards = []
# solve comb problem for tomorrow
super_arm = comb_learner.pull_super_arm()
img, axss = plt.subplots(nrows=2, ncols=3, figsize=(13, 6))
axs = axss.flatten()

for day in progressbar.progressbar(range(days)):
    if printBasicDebug:
        print(f"\n***** DAY {day} *****")
    users, products, campaigns, allocated_budget, prob_users = environment.get_core_entities()
    sim_obj = environment.play_one_day(N_user, reference_price, daily_budget,step_k, bool_alpha_noise,
                                       bool_n_noise)  # object with all the day info

    # disaggregated knapsack   --------------------------------------------
    rewards, available_budget = sim_obj["reward_k"]
    row_label_rewards, row_labels_dp_table, col_labels = table_metadata(len(products), len(users), available_budget)
    K = Knapsack(rewards=rewards, budgets=np.array(available_budget))
    K.init_for_pretty_print(row_labels=row_labels_dp_table, col_labels=col_labels)
    K.solve()

    arg_max = np.argmax(K.get_output()[0][-1])
    alloc = K.get_output()[1][-1][arg_max]
    reward = K.get_output()[0][-1][arg_max]
    rewards_disaggregated.append(reward)
    # -----------------------------------------------------------------------------
    # aggregated knapsack   --------------------------------------------
    rewards, available_budget = sim_obj["reward_k_agg"]
    row_label_rewards, row_labels_dp_table, col_labels = table_metadata(len(products), 1, available_budget)

    K = Knapsack(rewards=rewards, budgets=np.array(available_budget))
    K.init_for_pretty_print(row_labels=row_labels_dp_table, col_labels=col_labels)
    K.solve()
    arg_max = np.argmax(K.get_output()[0][-1])
    alloc = K.get_output()[1][-1][arg_max]
    reward = K.get_output()[0][-1][arg_max]
    # rewards_knapsack_agg.append(reward)

    # set knapsack solution on env
    k_alloc = alloc[1:]
    set_budgets_arm_env(k_alloc)
    # test result on env
    sim_obj_2 = environment.replicate_last_day(N_user,
                                               reference_price,
                                               bool_n_noise,
                                               bool_n_noise)

    rewards_knapsack_agg.append(sim_obj_2["profit_campaign"][-1] - np.sum(k_alloc))
    # -----------------------------------------------------------------

    # GTS and GPTS   --------------------------------------------
    # update with data from today for tomorrow
    set_budgets_arm_env(super_arm)
    # test result on env
    sim_obj_2 = environment.replicate_last_day(N_user,
                                               reference_price,
                                               bool_n_noise,
                                               bool_n_noise)

    profit_env = sim_obj_2["profit_campaign"][-1]
    learner_rewards.append(profit_env - np.sum(super_arm))
    profit_list = list(sim_obj_2["profit_campaign"][:-1])
    if boost_start and day <= 4:
        for i, s_arm in enumerate(super_arm):
            if s_arm == 0:
                # if a learner is pulling 0 give an high reward in an higher arm
                back_offset = np.random.randint(1, 4)
                forced_arm = np.sort(super_arm, axis=None)[-back_offset] # take random high arm value
                profit_list[i] = np.max(profit_list) * boost_discount + boost_bias
                super_arm[i] = forced_arm

    comb_learner.update_observations(super_arm, profit_list)
    # solve comb problem for tomorrow
    super_arm = comb_learner.pull_super_arm()
    # -----------------------------------------------------------------
    if printBasicDebug and day % 20 == 0:
        print(f"super arm:  {super_arm}")
        print(f"alloc knap: {alloc[1:]}")

    if day % 10 == 0:
        axs[5].cla()
        for i, rw in enumerate(rewards):
            axs[i].cla()
        """img, axss = plt.subplots(nrows=2, ncols=3, figsize=(13, 6))
        axs = axss.flatten()"""
    if day % 2 == 0:
        x = available_budget
        x2 = comb_learner.arms
        for i, rw in enumerate(rewards):
            axs[i].set_xlabel("budget")
            axs[i].set_ylabel("profit")
            axs[i].plot(x, rw)
            # axs[i].plot(x2, comb_learner.last_knapsack_reward[i])
            mean, std = comb_learner.get_gp_data()
            # print(std[0])
            # print(mean[0][0])
            axs[i].plot(x2, mean[i])
            axs[i].fill_between(
                np.array(x2).ravel(),
                mean[i] - 1.96 * std[i],
                mean[i] + 1.96 * std[i],
                alpha=0.5,
                label=r"95% confidence interval",
            )
        d = np.linspace(0, len(rewards_knapsack_agg), len(rewards_knapsack_agg))
        axs[5].set_xlabel("days")
        axs[5].set_ylabel("reward")
        axs[5].plot(d, rewards_knapsack_agg)
        axs[5].plot(d, learner_rewards)
        axs[5].plot(d, rewards_disaggregated)
        plt.pause(0.1)

# **** print profit functions and learned ones ***********
img, axss = plt.subplots(nrows=2, ncols=3, figsize=(13, 6))
axs = axss.flatten()
x = available_budget
x2 = comb_learner.arms
for i, rw in enumerate(rewards):
    axs[i].set_xlabel("budget")
    axs[i].set_ylabel("profit")
    axs[i].plot(x, rw)
    #axs[i].plot(x2, comb_learner.last_knapsack_reward[i])
    mean, std = comb_learner.get_gp_data()
    #print(std[0])
    #print(mean[0][0])
    axs[i].plot(x2, mean[i])
    axs[i].fill_between(
        np.array(x2).ravel(),
        mean[i] - 1.96 * std[i],
        mean[i] + 1.96 * std[i],
        alpha=0.5,
        label=r"95% confidence interval",
    )

plt.show()

# ********* statistical measures *********************
print(f"super arm:  {super_arm}")
print(f"alloc knap: {alloc[1:]}")

print(f"\n***** FINAL RESULT *****")
print(f"days simulated: {days}")
print(f"total profit:\t {sum(rewards_knapsack_agg):.4f}€")
print(f"standard deviation:\t {np.std(rewards_knapsack_agg):.4f}€")
print(f"Learner profit:\t {sum(learner_rewards):.4f}€")
print("----------------------------")
print(f"average profit:\t {np.mean(rewards_knapsack_agg):.4f}€")
print(f"\tstd:\t {np.std(rewards_knapsack_agg):.4f}€")
print(f"average reward:\t {np.mean(learner_rewards):.4f}€")
print(f"\tstd:\t {np.std(learner_rewards):.4f}€")
print(f"average regret\t {np.mean(np.array(rewards_knapsack_agg) - np.array(learner_rewards)):.4f}€")
print(f"\tstd:\t {np.std(np.array(rewards_knapsack_agg) - np.array(learner_rewards)):.4f}€")

plt.close()
d = np.linspace(0, len(rewards_knapsack_agg), len(rewards_knapsack_agg))

img, axss = plt.subplots(nrows=2, ncols=2, figsize=(13, 6))
axs = axss.flatten()

axs[0].set_xlabel("days")
axs[0].set_ylabel("reward")
axs[0].plot(d, rewards_knapsack_agg)
axs[0].plot(d, learner_rewards)

axs[1].set_xlabel("days")
axs[1].set_ylabel("cumulative reward")
axs[1].plot(d, np.cumsum(rewards_knapsack_agg))
axs[1].plot(d, np.cumsum(learner_rewards))

axs[2].set_xlabel("days")
axs[2].set_ylabel("cumulative regret")
axs[2].plot(d, np.cumsum(np.array(rewards_knapsack_agg) - np.array(learner_rewards)))

axs[3].set_xlabel("days")
axs[3].set_ylabel("regret")
axs[3].plot(d, np.array(rewards_knapsack_agg) - np.array(learner_rewards))
plt.show()

# TODO
#  1) graphs to show learned curve of profits
#  2) same simulation but comparing 2 algorithms
#  3) theoretical upper bound regret