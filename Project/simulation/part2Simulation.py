import numpy as np
import pandas as pd
from Knapsack import Knapsack
from simulation.Environment import Environment

""" @@@@ simulation SETUP @@@@ """
days = 1
N_user = 1000  # reference for what alpha = 1 refers to
reference_price = 3.5
step_knapsack = 5
n_budgets = 100
environment = Environment()
""" @@@@ ---------------- @@@@ """

for day in range(days):
    users, products, campaigns, allocated_budget, prob_users = environment.get_core_entities()
    sim_obj = environment.play_one_day(N_user, reference_price)  # object with all the day info
    profit1, profit2, profit3 = sim_obj["profit"]
    noise_alpha = sim_obj["noise"]

    # aggregated knapsack   --------------------------------------------
    rewards, available_budget = sim_obj["reward_k_agg"]
    row_label_rewards, row_labels_dp_table, col_labels = table_metadata(len(products), 1, available_budget)
    print("-" * 10 + " Independent rewards Table " + "-" * 10)
    print(pd.DataFrame(rewards, columns=col_labels, index=row_label_rewards))

    print("\n" + "*" * 25 + " Aggregated Knapsack execution " + "*" * 30 + "\n")
    K = Knapsack(rewards=rewards, budgets=np.array(available_budget))
    K.init_for_pretty_print(row_labels=row_labels_dp_table, col_labels=col_labels)
    K.solve()

    K.pretty_print_dp_table()  # prints the final dynamic programming table
    K.pretty_print_output(
        print_last_row_only=False)  # prints information about last row of the table, including allocations
    # -----------------------------------------------------------------

    # disaggregated knapsack   --------------------------------------------
    rewards, available_budget = sim_obj["reward_k"]
    row_label_rewards, row_labels_dp_table, col_labels = table_metadata(len(products), len(users), available_budget)
    print("-" * 10 + " Independent rewards Table " + "-" * 10)
    print(pd.DataFrame(rewards, columns=col_labels, index=row_label_rewards))

    print("\n" + "*" * 25 + " Disaggregated Knapsack execution " + "*" * 30 + "\n")
    K = Knapsack(rewards=rewards, budgets=np.array(available_budget))
    K.init_for_pretty_print(row_labels=row_labels_dp_table, col_labels=col_labels)
    K.solve()

    K.pretty_print_dp_table()  # prints the final dynamic programming table
    K.pretty_print_output(
        print_last_row_only=False)  # prints information about last row of the table, including allocations
    # -----------------------------------------------------------------

    # TODO allocate in the environment the budget found by knapsack
    # environment.set_campaign_budget()
    print(f"daily profits:\n\t u1:{profit1:.2f}€ || u2:{profit2:.2f}€ || u3:{profit3:.2f}€")


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
