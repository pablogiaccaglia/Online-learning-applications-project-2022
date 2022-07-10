import numpy as np
import pandas as pd
from Knapsack import Knapsack
from Part3.CombWrapper import CombWrapper
from Part3.GPUCB1_Learner import GPUCB1_Learner
from Part6.SwGPUCB1_Learner import SwGPUCB1_Learner
from Part6.CUSUM_GPUCB1_Learner import CusumGPUCB1Learner
from Part6.SwGTSLearner import SwGTSLearner
from simulation.Environment import Environment
import matplotlib.pyplot as plt

""" @@@@ simulation SETUP @@@@ """
days = 200
N_user = 200  # reference for what alpha = 1 refers to
reference_price = 4.0
daily_budget = 50 * 5
n_arms = 15
environment = Environment()

bool_alpha_noise = True
bool_n_noise = False
printBasicDebug = False
printKnapsackInfo = True
runAggregated = False  # mutual exclusive with run disaggregated
""" Change here the wrapper for the core bandit algorithm """
# gpucb1_learner = CombWrapper(GTS_Learner, 5, n_arms, daily_budget)
kwargs_cusum = {"samplesForRefPoint": 10,
                "epsilon":            0.05,
                "detectionThreshold": 10,
                "explorationAlpha":   0.01}
window_size = int(np.sqrt(days) + 0.1 * days)
kwargs_sw = {'window_size': window_size}

gpucb1_learner = CombWrapper(GPUCB1_Learner, 5, n_arms, daily_budget, is_ucb = True, is_gaussian = True)
sw_gpucb1_learner = CombWrapper(SwGPUCB1_Learner, 5, n_arms, daily_budget, is_ucb = True, kwargs = kwargs_sw, is_gaussian = True)
cusum_gpucb1_learner = CombWrapper(CusumGPUCB1Learner, 5, n_arms, daily_budget, is_ucb = True, kwargs = kwargs_cusum, is_gaussian = True)
sw_ts_learner = CombWrapper(SwGTSLearner, 5, n_arms, daily_budget, is_ucb = False, kwargs = kwargs_sw, is_gaussian = True)


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


rewards_knapsack_agg = []

gpucb1_rewards = []
sw_gpucb1_rewards = []
cusum_gpucb1_rewards = []
sw_ts_rewards = []

# solve comb problem for tomorrow
gpucb1_super_arm = gpucb1_learner.pull_super_arm()
sw_gpucb1_super_arm = sw_gpucb1_learner.pull_super_arm()
cusum_gpucb1_super_arm = cusum_gpucb1_learner.pull_super_arm()
sw_ts_super_arm = sw_ts_learner.pull_super_arm()

N_user_phases = [N_user, int(N_user - 0.5 * N_user), int(N_user + N_user), int(N_user - 0.6 * N_user), N_user]

n_phases = len(N_user_phases)

phase_size = days / n_phases

for day in range(days):

    current_phase = int(day / phase_size)

    N_user = N_user_phases[current_phase]

    if printBasicDebug:
        print(f"\n***** DAY {day} *****")
    users, products, campaigns, allocated_budget, prob_users = environment.get_core_entities()
    sim_obj = environment.play_one_day(N_user, reference_price, daily_budget, bool_alpha_noise,
                                       bool_n_noise)  # object with all the day info

    # aggregated knapsack   --------------------------------------------
    rewards, available_budget = sim_obj["reward_k_agg"]
    row_label_rewards, row_labels_dp_table, col_labels = table_metadata(len(products), 1, available_budget)

    K = Knapsack(rewards = rewards, budgets = np.array(available_budget))
    K.init_for_pretty_print(row_labels = row_labels_dp_table, col_labels = col_labels)
    K.solve()
    arg_max = np.argmax(K.get_output()[0][-1])
    alloc = K.get_output()[1][-1][arg_max]
    reward = K.get_output()[0][-1][arg_max]
    rewards_knapsack_agg.append(reward)

    if printBasicDebug:
        print("\n Aggregated")
        print(f"best allocation: {alloc}, total budget: {sum(alloc)}")
        print(f"reward: {reward}")
        set_budgets_knapsack_env(alloc)
        sim_obj_2 = environment.replicate_last_day(N_user,
                                                   reference_price,
                                                   bool_n_noise,
                                                   bool_n_noise)  # object with all the day info
        profit1, profit2, profit3, daily_profit = sim_obj_2["profit"]
        print(
                f"test allocation on env:\n\t || total:{daily_profit:.2f}€ || u1:{profit1:.2f}€ || u2:{profit2:.2f}€ || u3:{profit3:.2f}€")
        print("-" * 10 + " Independent rewards Table " + "-" * 10)
        print(pd.DataFrame(rewards, columns = col_labels, index = row_label_rewards))
        print("\n" + "*" * 25 + " Aggregated Knapsack execution " + "*" * 30 + "\n")
        K.pretty_print_dp_table()  # prints the final dynamic programming table
        """K.pretty_print_output(
            print_last_row_only=False)"""  # prints information about last row of the table, including allocations
    # -----------------------------------------------------------------

    # update with data from today for tomorrow
    set_budgets_arm_env(gpucb1_super_arm)
    # test result on env
    sim_obj_2 = environment.replicate_last_day(N_user,
                                               reference_price,
                                               bool_n_noise,
                                               bool_n_noise)

    gpucb1_learner.update_observations(gpucb1_super_arm, sim_obj_2["profit_campaign"][:-1])
    gpucb1_rewards.append(sim_obj_2["profit_campaign"][-1] - np.sum(gpucb1_super_arm))
    # solve comb problem for tomorrow
    gpucb1_super_arm = gpucb1_learner.pull_super_arm()

    # LEARNER 2

    set_budgets_arm_env(sw_gpucb1_super_arm)
    # test result on env
    sim_obj_3 = environment.replicate_last_day(N_user, reference_price, bool_n_noise, bool_n_noise)

    sw_gpucb1_learner.update_observations(sw_gpucb1_super_arm, sim_obj_3["profit_campaign"][:-1])
    sw_gpucb1_rewards.append(sim_obj_3["profit_campaign"][-1] - np.sum(sw_gpucb1_super_arm))
    # solve comb problem for tomorrow
    sw_gpucb1_super_arm = sw_gpucb1_learner.pull_super_arm()

    # LEARNER 3

    set_budgets_arm_env(cusum_gpucb1_super_arm)
    # test result on env
    sim_obj_4 = environment.replicate_last_day(N_user, reference_price, bool_n_noise, bool_n_noise)
    cusum_gpucb1_learner.update_observations(cusum_gpucb1_super_arm, sim_obj_4["profit_campaign"][:-1])
    cusum_gpucb1_rewards.append(sim_obj_4["profit_campaign"][-1] - np.sum(cusum_gpucb1_super_arm))
    # solve comb problem for tomorrow
    cusum_gpucb1_super_arm = cusum_gpucb1_learner.pull_super_arm()

    # LEARNER 4

    set_budgets_arm_env(sw_ts_super_arm)
    # test result on env
    sim_obj_5 = environment.replicate_last_day(N_user, reference_price, bool_n_noise, bool_n_noise)
    sw_ts_learner.update_observations(sw_ts_super_arm, sim_obj_4["profit_campaign"][:-1])
    sw_ts_rewards.append(sim_obj_5["profit_campaign"][-1] - np.sum(sw_ts_super_arm))
    # solve comb problem for tomorrow
    sw_ts_super_arm = cusum_gpucb1_learner.pull_super_arm()

    # -----------------------------------------------------------------
    if day % 20 == 0:
        print(f"super arm 1:  {gpucb1_super_arm}")
        print(f"alloc knap: {alloc[1:]}")

        print(f"super arm 2:  {sw_gpucb1_super_arm}")
        print(f"alloc knap: {alloc[1:]}")

        print(f"super arm 3:  {cusum_gpucb1_super_arm}")
        print(f"alloc knap: {alloc[1:]}")

        print(f"super arm 4:  {sw_ts_super_arm}")
        print(f"alloc knap: {alloc[1:]}")

print(f"super arm:  {gpucb1_super_arm}")
print(f"alloc knap: {alloc[1:]}")

print(f"super arm2:  {sw_gpucb1_super_arm}")
print(f"alloc knap: {alloc[1:]}")

print(f"super arm3:  {cusum_gpucb1_super_arm}")
print(f"alloc knap: {alloc[1:]}")

print(f"super arm4:  {sw_ts_super_arm}")
print(f"alloc knap: {alloc[1:]}")

print(f"\n***** FINAL RESULT LEARNER GP-UCB1*****")
print(f"days simulated: {days}")
print(f"total profit:\t {sum(rewards_knapsack_agg):.4f}€")
print(f"standard deviation:\t {np.std(rewards_knapsack_agg):.4f}€")
print(f"Learner profit:\t {sum(gpucb1_rewards):.4f}€")
print("----------------------------")
print(f"average profit:\t {np.mean(rewards_knapsack_agg):.4f}€")
print(f"\tstd:\t {np.std(rewards_knapsack_agg):.4f}€")
print(f"average reward:\t {np.mean(gpucb1_rewards):.4f}€")
print(f"\tstd:\t {np.std(gpucb1_rewards):.4f}€")
print(f"average regret\t {np.mean(np.array(rewards_knapsack_agg) - np.array(gpucb1_rewards)):.4f}€")
print(f"\tstd:\t {np.std(np.array(rewards_knapsack_agg) - np.array(gpucb1_rewards)):.4f}€")

print(f"\n***** FINAL RESULT SW-GP-UCB1*****")
print(f"days simulated: {days}")
print(f"total profit:\t {sum(rewards_knapsack_agg):.4f}€")
print(f"standard deviation:\t {np.std(rewards_knapsack_agg):.4f}€")
print(f"Learner profit:\t {sum(sw_gpucb1_rewards):.4f}€")
print("----------------------------")
print(f"average profit:\t {np.mean(rewards_knapsack_agg):.4f}€")
print(f"\tstd:\t {np.std(rewards_knapsack_agg):.4f}€")
print(f"average reward:\t {np.mean(sw_gpucb1_rewards):.4f}€")
print(f"\tstd:\t {np.std(sw_gpucb1_rewards):.4f}€")
print(f"average regret\t {np.mean(np.array(rewards_knapsack_agg) - np.array(sw_gpucb1_rewards)):.4f}€")
print(f"\tstd:\t {np.std(np.array(rewards_knapsack_agg) - np.array(sw_gpucb1_rewards)):.4f}€")

print(f"\n***** FINAL RESULT CUSUM-GP-UCB1*****")
print(f"days simulated: {days}")
print(f"total profit:\t {sum(rewards_knapsack_agg):.4f}€")
print(f"standard deviation:\t {np.std(rewards_knapsack_agg):.4f}€")
print(f"Learner profit:\t {sum(cusum_gpucb1_rewards):.4f}€")
print("----------------------------")
print(f"average profit:\t {np.mean(rewards_knapsack_agg):.4f}€")
print(f"\tstd:\t {np.std(rewards_knapsack_agg):.4f}€")
print(f"average reward:\t {np.mean(cusum_gpucb1_rewards):.4f}€")
print(f"\tstd:\t {np.std(cusum_gpucb1_rewards):.4f}€")
print(f"average regret\t {np.mean(np.array(rewards_knapsack_agg) - np.array(cusum_gpucb1_rewards)):.4f}€")
print(f"\tstd:\t {np.std(np.array(rewards_knapsack_agg) - np.array(cusum_gpucb1_rewards)):.4f}€")

print(f"\n***** FINAL RESULT SW-TS*****")
print(f"days simulated: {days}")
print(f"total profit:\t {sum(rewards_knapsack_agg):.4f}€")
print(f"standard deviation:\t {np.std(rewards_knapsack_agg):.4f}€")
print(f"Learner profit:\t {sum(sw_ts_rewards):.4f}€")
print("----------------------------")
print(f"average profit:\t {np.mean(rewards_knapsack_agg):.4f}€")
print(f"\tstd:\t {np.std(rewards_knapsack_agg):.4f}€")
print(f"average reward:\t {np.mean(sw_ts_rewards):.4f}€")
print(f"\tstd:\t {np.std(sw_ts_rewards):.4f}€")
print(f"average regret\t {np.mean(np.array(rewards_knapsack_agg) - np.array(sw_ts_rewards)):.4f}€")
print(f"\tstd:\t {np.std(np.array(rewards_knapsack_agg) - np.array(sw_ts_rewards)):.4f}€")

plt.close()
d = np.linspace(0, len(rewards_knapsack_agg), len(rewards_knapsack_agg))

img, axss = plt.subplots(nrows = 2, ncols = 2, figsize = (13, 6))
axs = axss.flatten()

axs[0].set_xlabel("days")
axs[0].set_ylabel("reward")
axs[0].plot(d, rewards_knapsack_agg, 'g', label = "clairvoyant")
axs[0].plot(d, gpucb1_rewards, 'r', label = "GP-UCB1")
axs[0].plot(d, sw_gpucb1_rewards, 'b', label = "SW-GP-UCB1")
axs[0].plot(d, cusum_gpucb1_rewards, 'c', label = "CUSUM-GP-UCB1")
axs[0].plot(d, sw_ts_rewards, 'm', label = "SW_TS")
axs[0].legend(loc = "upper left")

axs[1].set_xlabel("days")
axs[1].set_ylabel("cumulative reward")
axs[1].plot(d, np.cumsum(rewards_knapsack_agg), 'g', label = "clairvoyant")
axs[1].plot(d, np.cumsum(gpucb1_rewards), 'r', label = "GP-UCB1")
axs[1].plot(d, np.cumsum(sw_gpucb1_rewards), 'b', label = "SW-GP-UCB1")
axs[1].plot(d, np.cumsum(cusum_gpucb1_rewards), 'c', label = "CUSUM-GP-UCB1")
axs[1].plot(d, np.cumsum(sw_ts_rewards), 'm', label = "SW_TS")
axs[1].legend(loc = "upper left")

axs[2].set_xlabel("days")
axs[2].set_ylabel("cumulative regret")
axs[2].plot(d, np.cumsum(np.array(rewards_knapsack_agg) - np.array(gpucb1_rewards)), 'r', label = "GP-UCB1")
axs[2].plot(d, np.cumsum(np.array(rewards_knapsack_agg) - np.array(sw_gpucb1_rewards)), 'b', label = "SW-GP-UCB1")
axs[2].plot(d, np.cumsum(np.array(rewards_knapsack_agg) - np.array(cusum_gpucb1_rewards)), 'c', label = "CUSUM-GP-UCB1")
axs[2].plot(d, np.cumsum(np.array(rewards_knapsack_agg) - np.array(sw_ts_rewards)), 'm', label = "SW_TS")
axs[2].legend(loc = "upper left")

axs[3].set_xlabel("days")
axs[3].set_ylabel("regret")
axs[3].plot(d, np.array(rewards_knapsack_agg) - np.array(gpucb1_rewards), 'r', label = "GP-UCB1")
axs[3].plot(d, np.array(rewards_knapsack_agg) - np.array(sw_gpucb1_rewards), 'b', label = "SW-GP-UCB1")
axs[3].plot(d, np.array(rewards_knapsack_agg) - np.array(cusum_gpucb1_rewards), 'c', label = "CUSUM-GP-UCB1")
axs[3].plot(d, np.array(rewards_knapsack_agg) - np.array(sw_ts_rewards), 'm', label = "SW_TS")
axs[3].legend(loc = "upper left")

plt.show()

# TODO
#  1) graphs to show learned curve of profits
#  2) same simulation but comparing 2 algorithms
#  3) theoretical upper bound regret
