import numpy as np
import pandas as pd
from Knapsack import Knapsack
from simulation.Environment import Environment
import matplotlib.pyplot as plt
import progressbar


""" @@@@ simulation SETUP @@@@ """
days = 6
N_user = 1000  # reference for what alpha = 1 refers to
reference_price = 3.5
daily_budget = 800
step_k=2
environment = Environment()

bool_alpha_noise = False
bool_n_noise = False
printBasicDebug = False
printKnapsackInfo = False
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


def set_budgets_env(knapsack_alloc):
    for i, b in enumerate(knapsack_alloc[1:]):
        environment.set_campaign_budget(i, b)


rewards_aggregated = []
rewards_disaggregated = []
for day in progressbar.progressbar(range(days)):
    if printBasicDebug:
        print(f"\n***** DAY {day} *****")
    users, products, campaigns, allocated_budget, prob_users = environment.get_core_entities()
    sim_obj = environment.play_one_day(N_user, reference_price, daily_budget,step_k , bool_alpha_noise,
                                       bool_n_noise)  # object with all the day info
    profit1, profit2, profit3, daily_profit = sim_obj["profit"]
    p_cmp_1, p_cmp_2, p_cmp_3, p_cmp_4, p_cmp_5, tot_camp = sim_obj["profit_campaign"]
    noise_alpha = sim_obj["noise"]

    # aggregated knapsack   --------------------------------------------
    rewards, available_budget = sim_obj["reward_k_agg"]
    row_label_rewards, row_labels_dp_table, col_labels = table_metadata(len(products), 1, available_budget)
    K = Knapsack(rewards=rewards, budgets=np.array(available_budget))
    K.init_for_pretty_print(row_labels=row_labels_dp_table, col_labels=col_labels)
    K.solve()

    arg_max = np.argmax(K.get_output()[0][-1])
    alloc = K.get_output()[1][-1][arg_max]
    reward = K.get_output()[0][-1][arg_max]
    rewards_aggregated.append(reward)

    print(f"best allocation: {alloc}, total budget: {sum(alloc)}")
    print(f"reward: {reward}")
    set_budgets_env(alloc)
    sim_obj_2 = environment.replicate_last_day(N_user,
                                               reference_price,
                                               bool_n_noise,
                                               bool_n_noise)  # object with all the day info
    profit1, profit2, profit3, daily_profit = sim_obj_2["profit"]
    print(
        f"test allocation on env:\n\t || total:{daily_profit:.2f}€ || u1:{profit1:.2f}€ || u2:{profit2:.2f}€ || u3:{profit3:.2f}€")

    if printBasicDebug:
        print("\n Aggregated")
        print(f"best allocation: {alloc}, total budget: {sum(alloc)}")
        print(f"reward: {reward}")
        set_budgets_env(alloc)
        sim_obj_2 = environment.replicate_last_day(N_user,
                                                   reference_price,
                                                   bool_n_noise,
                                                   bool_n_noise)  # object with all the day info
        profit1, profit2, profit3, daily_profit = sim_obj_2["profit"]
        print(
            f"test allocation on env:\n\t || total:{daily_profit:.2f}€ || u1:{profit1:.2f}€ || u2:{profit2:.2f}€ || u3:{profit3:.2f}€")
    print("-" * 10 + " Independent rewards Table " + "-" * 10)
    print(pd.DataFrame(rewards, columns=col_labels, index=row_label_rewards))
    if printKnapsackInfo:
        print("-" * 10 + " Independent rewards Table " + "-" * 10)
        print(pd.DataFrame(rewards, columns=col_labels, index=row_label_rewards))
        print("\n" + "*" * 25 + " Aggregated Knapsack execution " + "*" * 30 + "\n")
        K.pretty_print_dp_table()  # prints the final dynamic programming table
        K.pretty_print_output(
            print_last_row_only=False)  # prints information about last row of the table, including allocations
    # -----------------------------------------------------------------

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
    print("-" * 10 + " Independent rewards Table " + "-" * 10)
    print(pd.DataFrame(rewards, columns=col_labels, index=row_label_rewards))
    if printBasicDebug:
        alloc = K.get_output()[1][-1][-1]
        reward = K.get_output()[0][-1][-1]
        print("\n Disaggregated")
        print(f"best allocation: {alloc}, total budget: {sum(alloc)}")
        print(f"reward: {reward}")

    if printKnapsackInfo:
        print("-" * 10 + " Independent rewards Table " + "-" * 10)
        print(pd.DataFrame(rewards, columns=col_labels, index=row_label_rewards))
        print("\n" + "*" * 25 + " Disaggregated Knapsack execution " + "*" * 30 + "\n")
        K.pretty_print_dp_table()  # prints the final dynamic programming table
        K.pretty_print_output(
            print_last_row_only=False)  # prints information about last row of the table, including allocations
    # -----------------------------------------------------------------
print(f"\n***** FINAL RESULT *****")
print(f"aggregated profit:\t {sum(rewards_aggregated):.2f}€")
print(f"\taverage:\t {np.mean(rewards_aggregated):.2f}€")
print(f"\tstd:\t\t {np.std(rewards_aggregated):.2f}€")
print("---------------------------")
print(f"disaggregated profit:\t {sum(rewards_disaggregated):.2f}€")
print(f"\taverage:\t {np.mean(rewards_disaggregated):.2f}€")
print(f"\tstd:\t\t {np.std(rewards_disaggregated):.2f}€")
print("---------------------------")
print(f"Profit ratio: {sum(rewards_aggregated)/sum(rewards_disaggregated)}")
print(f"total regret:\t {np.mean(rewards_aggregated):.2f}€")
print(f"\taverage\t {np.mean(np.array(rewards_disaggregated) - np.array(rewards_aggregated)):.2f}€")
print(f"\tstd:\t {np.std(np.array(rewards_disaggregated) - np.array(rewards_aggregated)):.2f}€")

plt.close()
days = np.linspace(0, len(rewards_aggregated), len(rewards_aggregated))

img, axss = plt.subplots(nrows=1, ncols=2, figsize=(13, 6))
axs = axss.flatten()

axs[0].set_xlabel("days")
axs[0].set_ylabel("reward")
axs[0].plot(days, rewards_aggregated, label='aggregated')
axs[0].plot(days, rewards_disaggregated, label='disaggregated')
axs[0].legend()

axs[1].set_xlabel("days")
axs[1].set_ylabel("cumulative reward")
axs[1].plot(days, np.cumsum(rewards_aggregated), label='aggregated')
axs[1].plot(days, np.cumsum(rewards_disaggregated), label='disaggregated')
axs[1].legend()
plt.show()
