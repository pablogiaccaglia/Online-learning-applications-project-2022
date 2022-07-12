import numpy as np
import pandas as pd

from OfflineWeightsLearner import OfflineWeightsLearner
from Knapsack import Knapsack
from OnlineWeightsLearner import OnlineWeightsLearner
from Part3.GTS_Learner import GTS_Learner
from Part3.GPTS_Learner import GPTS_Learner
from Part3.CombWrapper import CombWrapper
from Part3.GPTS_Learner import GPTS_Learner
from simulation.Environment import Environment
import matplotlib.pyplot as plt
import Utils as util

""" @@@@ simulation SETUP @@@@ """
days = 2
N_user = 300  # reference for what alpha = 1 refers to
reference_price = 2.0
daily_budget = 500
n_arms = 100
environment = Environment()

bool_alpha_noise = True
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
Run montecarlo for each user and return :
    -> estimation-fully-connected
    -> estimation-2-neighbours
    -> true graph
"""


def run_graph_estimate(users,
                       numOfEpisodes=100,
                       simulations=100,
                       delta=0.2,  # higher delta, fewer simulations
                       epsilon=0.1,
                       seeds=1,
                       silent=True):
    true_result_history = []
    estimation_fully_con = []
    estimation_2_neigh = []

    for user in users:
        """ Graph weights estimation simualation - user 1 - OFFLINE LEARNING - fully connected graph"""
        adjacency_matrix = np.array(user.weighted_graph.get_adjacency_matrix())
        n_campaigns = len(campaigns)

        estimatedProbs = OfflineWeightsLearner.estimateProbabilities(numOfEpisodes=numOfEpisodes,
                                                                     targetNodes=[i for i in range(n_campaigns)],
                                                                     numberOfNodes=n_campaigns,
                                                                     probabilitiesMatrix=adjacency_matrix)

        if not silent:
            print("\n-" * 10 + " Weights estimation - OFFLINE - fully connected graph " + "-" * 10)
            print("\nTrue Probability Matrix: \n", util.get_prettyprint_array(adjacency_matrix))
            print("\nEstimated Probability Matrix: \n", util.get_prettyprint_array(estimatedProbs))

        """ Graph weights estimation simualation - user 1 - ONLINE LEARNING - fully connected graph"""

        """Assign a beta to each edge and learning happens from influence episodes.
            Influence starts from a single seed, chosen through the greedy algorithm as the 
            one with the highest influence"""

        if not silent:
            print("\n-" * 10 + " Weights estimation - ONLINE - fully connected graph " + "-" * 10)

        # deltas = [0.95, 0.8, 0.4, 0.2]
        monte_carlo_repetitions = int((1 / (epsilon ** 2)) * np.log(seeds + 1) * np.log(1 / delta))
        if not silent:
            print("\nmonte carlo simulations: " + str(monte_carlo_repetitions))
        plt.close()
        estimatedGraph = OnlineWeightsLearner.estimate_weights(true_graph=user.weighted_graph,
                                                               simulations=simulations,
                                                               monte_carlo_repetitions=monte_carlo_repetitions)
        estimation_fully_con.append(estimatedGraph)
        if not silent:
            print("\nTrue Probability Matrix: \n",
                  util.get_prettyprint_array(user.weighted_graph.get_adjacency_matrix()))
            print("\nEstimated Probability Matrix: \n",
                  util.get_prettyprint_array(estimatedGraph.get_adjacency_matrix()))

            print("\nBetas Matrix: \n", util.get_prettyprint_array(estimatedGraph.get_betas_matrix()))

        """ Graph weights estimation simualation - user 1 - ONLINE LEARNING - graph with 2 neighbours per node"""

        """Assign a beta to each edge and learning happens from influence episodes.
            Influence starts from a single seed, chosen through the greedy algorithm as the 
            one with the highest influence"""

        if not silent:
            print("\n-" * 10 + " Weights estimation - ONLINE - graph with 2 neighbours per node " + "-" * 10)
        monte_carlo_repetitions = int((1 / (epsilon ** 2)) * np.log(seeds + 1) * np.log(1 / delta))
        if not silent:
            print("\nmonte carlo simulations: " + str(monte_carlo_repetitions))
        plt.close()

        ecommerceGraph = util.get_ecommerce_graph(products=products)
        estimatedGraph = OnlineWeightsLearner.estimate_weights(true_graph=ecommerceGraph,
                                                               simulations=simulations,
                                                               monte_carlo_repetitions=monte_carlo_repetitions)
        estimation_2_neigh.append(estimatedGraph)
        true_result_history.append(ecommerceGraph)
        if not silent:
            print("\nTrue Probability Matrix: \n", util.get_prettyprint_array(ecommerceGraph.get_adjacency_matrix()))
            print("\nEstimated Probability Matrix: \n",
                  util.get_prettyprint_array(estimatedGraph.get_adjacency_matrix()))
            print("\nBetas Matrix: \n", util.get_prettyprint_array(estimatedGraph.get_betas_matrix()))
    return estimation_fully_con, estimation_2_neigh, true_result_history


#   **** MONTECARLO EXECUTION BEFORE SIMULATION ****
users, products, campaigns, allocated_budget, prob_users = environment.get_core_entities()
estimated_weights = run_graph_estimate(users=users)
print(estimated_weights)
#   ************************************************

rewards_knapsack_agg = []
learner_rewards = []
# solve comb problem for tomorrow
super_arm = comb_learner.pull_super_arm()

for day in range(days):
    if printBasicDebug:
        print(f"\n***** DAY {day} *****")
    users, products, campaigns, allocated_budget, prob_users = environment.get_core_entities()
    sim_obj = environment.play_one_day(N_user, reference_price, daily_budget, bool_alpha_noise,
                                       bool_n_noise)  # object with all the day info

    # aggregated knapsack   --------------------------------------------
    rewards, available_budget = sim_obj["reward_k_agg"]
    row_label_rewards, row_labels_dp_table, col_labels = table_metadata(len(products), 1, available_budget)

    K = Knapsack(rewards=rewards, budgets=np.array(available_budget))
    K.init_for_pretty_print(row_labels=row_labels_dp_table, col_labels=col_labels)
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
        print(pd.DataFrame(rewards, columns=col_labels, index=row_label_rewards))
        print("\n" + "*" * 25 + " Aggregated Knapsack execution " + "*" * 30 + "\n")
        K.pretty_print_dp_table()  # prints the final dynamic programming table
        """K.pretty_print_output(
            print_last_row_only=False)"""  # prints information about last row of the table, including allocations
    # -----------------------------------------------------------------

    # GTS and GPTS   --------------------------------------------
    # update with data from today for tomorrow
    set_budgets_arm_env(super_arm)
    # test result on env
    sim_obj_2 = environment.replicate_last_day(N_user,
                                               reference_price,
                                               bool_n_noise,
                                               bool_n_noise)

    comb_learner.update_observations(super_arm, sim_obj_2["profit_campaign"][:-1])
    learner_rewards.append(sim_obj_2["profit_campaign"][-1] - np.sum(super_arm))
    # solve comb problem for tomorrow
    super_arm = comb_learner.pull_super_arm()
    # -----------------------------------------------------------------
    if day % 20 == 0:
        print(f"super arm:  {super_arm}")
        print(f"alloc knap: {alloc[1:]}")

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
#  1) decide where and how often to run montecarlo weight learning
#  2) implement procedure to use learned graph in users during bandit for advertising,
#  keeping knapsack working on real graph
#  3) make printing of MC nicer, now there is an flag to set it to silent of full verbose
