import numpy as np
import matplotlib.pyplot as plt
import Project.Utils as util
from Project.Campaign import Campaign
from Project.Product import Product
from Project.User import User
from GTS_Learner import GTS_Learner
from GPTS_Learner import GPTS_Learner
from GP_Campaign import GP_Campaign
from Bidding_Environment import Bidding_Environment
from Project.Knapsack import Knapsack

""" Products SETUP """

prod1 = Product(1, 0.50, secondary_list=[2, 3])
prod2 = Product(2, 0.625, secondary_list=[3, 4])
prod3 = Product(3, 0.75, secondary_list=[1, 5])
prod4 = Product(4, 0.875, secondary_list=[2, 5])
prod5 = Product(5, 1.00, secondary_list=[1, 4])

products = [prod1, prod2, prod3, prod4, prod5]

""" Alpha functions """
mv = 1  # don't change it,  max alpha function value
act = 20  # activation
alpha_usr1 = [
    util.new_alpha_function(saturation_speed=0.025, max_value=mv, activation=act),
    util.new_alpha_function(saturation_speed=0.06, max_value=mv, activation=act),
    util.new_alpha_function(saturation_speed=0.03, max_value=mv, activation=act),
    util.new_alpha_function(saturation_speed=0.04, max_value=mv, activation=act),
    util.new_alpha_function(saturation_speed=0.04, max_value=mv, activation=act),
    util.new_alpha_function(saturation_speed=0.06, max_value=mv, activation=act)
]
act = 25  # activation
alpha_usr2 = [
    util.new_alpha_function(saturation_speed=0.18, max_value=mv, activation=act),
    util.new_alpha_function(saturation_speed=0.16, max_value=mv, activation=act),
    util.new_alpha_function(saturation_speed=0.23, max_value=mv, activation=act),
    util.new_alpha_function(saturation_speed=0.25, max_value=mv, activation=act),
    util.new_alpha_function(saturation_speed=0.21, max_value=mv, activation=act),
    util.new_alpha_function(saturation_speed=0.19, max_value=mv, activation=act)
]
act = 15  # activation
alpha_usr3 = [
    util.new_alpha_function(saturation_speed=0.08, max_value=mv, activation=act),
    util.new_alpha_function(saturation_speed=0.12, max_value=mv, activation=act),
    util.new_alpha_function(saturation_speed=0.07, max_value=mv, activation=act),
    util.new_alpha_function(saturation_speed=0.10, max_value=mv, activation=act),
    util.new_alpha_function(saturation_speed=0.07, max_value=mv, activation=act),
    util.new_alpha_function(saturation_speed=0.09, max_value=mv, activation=act)
]

""" Users SETUP """

prob_user1 = 0.25
prob_user2 = 0.45
prob_user3 = 0.30

prob_users = [prob_user1, prob_user2, prob_user3]

# alpha_users = [alpha_usr1,alpha_usr2,alpha_usr3]
# alpha_agg = []
# for i in range(len(alpha_usr1)):
#     alpha = 0
#     for j in range(len(alpha_users)):
#         alpha += alpha_users[i][j] * prob_users[j]
#     alpha_agg.append(alpha)
#
# test_alpha = alpha_agg
# img, axss = plt.subplots(nrows=2, ncols=3, figsize=(13, 6))
# axs = axss.flatten()
# for i in range(6):
#     x = np.linspace(0, 100, 100)
#     y = test_alpha[i](x).clip(0.0)  # visual clip, be careful using them plain
#     axs[i].set_xlabel("budget")
#     axs[i].set_ylabel(f"alpha{i}_val")
#     axs[i].plot(x, y)
#
# if True:  # show or not alpha plot
#     plt.show()


# base         [0.50, 0.625, 0.75, 0.875, 1.00]
res_prices_1 = [0.60, 0.725, 0.65, 0.975, 1.10]
res_prices_2 = [0.60, 0.525, 0.85, 0.775, 1.10]
res_prices_3 = [0.60, 0.725, 0.85, 0.975, 1.10]

exp_number_purchase_1 = [2.0, 2.0, 2.0, 2.0, 3.5]  # 2   - 3.5
exp_number_purchase_2 = [1.0, 1.0, 1.0, 1.0, 1.5]  # 1   - 1.5
exp_number_purchase_3 = [1.5, 1.5, 1.5, 1.5, 2.0]  # 1.5 - 2

user1 = User(id=1,
             reservation_prices=res_prices_1,
             lmbda=0.8,
             alpha_functions=alpha_usr1,
             exp_number_purchase=exp_number_purchase_1,
             weighted_graph=util.random_fully_connected_graph(products)
             )

user2 = User(id=2,
             reservation_prices=res_prices_2,
             lmbda=0.5,
             alpha_functions=alpha_usr2,
             exp_number_purchase=exp_number_purchase_2,
             weighted_graph=util.random_fully_connected_graph(products)
             )

user3 = User(id=3,
             reservation_prices=res_prices_3,
             lmbda=0.65,
             alpha_functions=alpha_usr3,
             exp_number_purchase=exp_number_purchase_3,
             weighted_graph=util.random_fully_connected_graph(products),
             )

users = [user1, user2, user3]

""" Campaigns SETUP """
alpha_i_max = [0.5, 0.5, 0.3, 0.4, 0.3]
allocated_budget = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # range 0-100
competitor_alpha = np.sum(alpha_i_max)

cmp1 = Campaign(1, allocated_budget[0], alpha_i_max=alpha_i_max[0])
cmp2 = Campaign(2, allocated_budget[1], alpha_i_max=alpha_i_max[1])
cmp3 = Campaign(3, allocated_budget[2], alpha_i_max=alpha_i_max[2])
cmp4 = Campaign(4, allocated_budget[3], alpha_i_max=alpha_i_max[3])
cmp5 = Campaign(5, allocated_budget[4], alpha_i_max=alpha_i_max[4])

campaigns = [cmp1, cmp2, cmp3, cmp4, cmp5]

""" Multiple days simulation """

days = 21
N_user = 1000  # reference for what alpha= 1 refers to
reference_price = 3.5

""" Some constants """

N_CLASSES = len(users)
N_CAMPAIGNS = len(products)
N_BUDGETS = len(allocated_budget)

print()

rewards_sim = np.zeros((N_CAMPAIGNS, N_BUDGETS), dtype=np.single)

for cmp_index in range(0, N_CAMPAIGNS):

    for budget_idx in range(0, N_BUDGETS):

        """ Each campaign is observed with all the budget values """
        cmp = Campaign(id=cmp_index + 1,
                       allocated_budget=allocated_budget[budget_idx],
                       alpha_i_max=alpha_i_max[cmp_index])
        expected_gross_profit = 0.0
        for user_idx, user in enumerate(users):
            prob_user = prob_users[user_idx]
            alpha = cmp.get_alpha_i(user.alpha_functions[cmp_index])
            value_per_click = user.expected_profit()[cmp_index]
            expected_gross_profit += prob_user * alpha * value_per_click
        rewards_sim[cmp_index][budget_idx] = np.single(expected_gross_profit)


rewards_sim = reference_price * N_user * rewards_sim  # convert the pure number rewards in euros

# todo: the actual simulation starts here
"""
PART 3 SIMULATION
"""

T = 30
n_experiments = 5
gts_rewards_per_experiment = []
gpts_rewards_per_experiment = []
env = Bidding_Environment(users=users, campaigns=campaigns, prob_users=prob_users)
for e in range(0, n_experiments):

    #env = Bidding_Environment(users=users, campaigns=campaigns, prob_users=prob_users)

    gts_learner = GTS_Learner(arms=allocated_budget, n_campaigns=N_CAMPAIGNS)
    gpts_learner = GPTS_Learner(arms=allocated_budget, n_campaigns=N_CAMPAIGNS)

    for t in range(0, T):
        # Gaussian Thompson Sampling
        pulled_arm = gts_learner.pull_arm()
        rewards_gts = env.get_aggregated_profit(pulled_arm)
        rewards_gts_euros = [r*N_user*reference_price for r in rewards_gts] # todo: ask others if reference prices should differ
        gts_learner.update(pulled_arm, rewards_gts_euros)

        # GP Thompson Sampling
        pulled_arm = gpts_learner.pull_arm()
        rewards_gpts = env.get_aggregated_profit(pulled_arm)
        rewards_gpts_euros = [r*N_user*reference_price for r in rewards_gpts]
        gpts_learner.update(pulled_arm, rewards_gpts_euros)

    gts_rewards_per_experiment.append(gts_learner.collected_rewards.sum(axis=1))
    gpts_rewards_per_experiment.append(gpts_learner.collected_rewards.sum(axis=1))

"""
compute clairvoyant here, it will change when we will add noise
"""
K = Knapsack(rewards=rewards_sim, budgets=np.array(allocated_budget))
K.solve()
opt_arm = K.allocations[-1][-1][1:]
opt = env.get_aggregated_profit(opt_arm)
opt_euros = [o*N_user*reference_price for o in opt]

plt.figure(0)
plt.ylabel('Regret')
plt.xlabel('t')

# gts_rewards_per_experiment = np.array(gts_rewards_per_experiment).sum(axis=1)
# gpts_rewards_per_experiment = np.array(gpts_rewards_per_experiment).sum(axis=1)
opt_euros = np.array(opt_euros).sum()
plt.plot(np.cumsum(np.mean(opt_euros - gts_rewards_per_experiment, axis=0)), 'r')
plt.plot(np.cumsum(np.mean(opt_euros - gpts_rewards_per_experiment, axis=0)), 'g')
#aaa = np.mean(opt - gts_rewards_per_experiment, axis=0)
plt.legend(['GTS', 'GPTS'])
plt.show()
