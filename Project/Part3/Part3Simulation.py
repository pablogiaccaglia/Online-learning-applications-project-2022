import numpy as np
import matplotlib.pyplot as plt
import Utils as util
from Campaign import Campaign
from Product import Product
from User import User
from GTS_Learner import GTS_Learner
from GPTS_Learner import GPTS_Learner
from GP_Campaign import GP_Campaign
from Bidding_Environment import Bidding_Environment
from Knapsack import Knapsack


""" Multiple days simulation """

days = 21
N_user = 1000  # reference for what alpha= 1 refers to
reference_price = 3.5

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
            value_per_click = user.expected_profit(np.ones(5))[cmp_index]
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
#env = Bidding_Environment(users=users, campaigns=campaigns, prob_users=prob_users)
for e in range(0, n_experiments):

    # env = Bidding_Environment(users=users, campaigns=campaigns, prob_users=prob_users)

    gts_learner = GTS_Learner(arms=allocated_budget, n_campaigns=N_CAMPAIGNS)
    gpts_learner = GPTS_Learner(arms=allocated_budget, n_campaigns=N_CAMPAIGNS)

    for t in range(0, T):
        # Gaussian Thompson Sampling
        pulled_arm = gts_learner.pull_arm()
        rewards_gts = env.get_aggregated_profit(pulled_arm)
        rewards_gts_euros = [r * N_user * reference_price for r in
                             rewards_gts]  # todo: ask others if reference prices should differ
        gts_learner.update(pulled_arm, rewards_gts_euros)

        # GP Thompson Sampling
        pulled_arm = gpts_learner.pull_arm()
        rewards_gpts = env.get_aggregated_profit(pulled_arm)
        rewards_gpts_euros = [r * N_user * reference_price for r in rewards_gpts]
        gpts_learner.update(pulled_arm, rewards_gpts_euros)

    gts_rewards_per_experiment.append(gts_learner.collected_rewards.sum(axis=1))
    gpts_rewards_per_experiment.append(gpts_learner.collected_rewards.sum(axis=1))

"""
compute clairvoyant here, it will change when we will add noise
"""
"""K = Knapsack(rewards=rewards_sim, budgets=np.array(allocated_budget))
K.solve()
opt_arm = K.allocations[-1][-1][1:]
opt = env.get_aggregated_profit(opt_arm)
opt_euros = [o * N_user * reference_price for o in opt]"""

plt.figure(0)
plt.ylabel('Regret')
plt.xlabel('t')

# gts_rewards_per_experiment = np.array(gts_rewards_per_experiment).sum(axis=1)
# gpts_rewards_per_experiment = np.array(gpts_rewards_per_experiment).sum(axis=1)
opt_euros = np.array(opt_euros).sum()
plt.plot(np.cumsum(np.mean(opt_euros - gts_rewards_per_experiment, axis=0)), 'r')
plt.plot(np.cumsum(np.mean(opt_euros - gpts_rewards_per_experiment, axis=0)), 'g')
# aaa = np.mean(opt - gts_rewards_per_experiment, axis=0)
plt.legend(['GTS', 'GPTS'])
plt.show()
