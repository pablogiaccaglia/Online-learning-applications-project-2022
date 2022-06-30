import numpy as np
import matplotlib.pyplot as plt
import Utils as util
from Campaign import Campaign
from Product import Product
from User import User
import pandas as pd
from Knapsack import Knapsack

""" Products SETUP """

prod1 = Product(1, 0.50, secondary_list = [2, 3])
prod2 = Product(2, 0.625, secondary_list = [3, 4])
prod3 = Product(3, 0.75, secondary_list = [1, 5])
prod4 = Product(4, 0.875, secondary_list = [2, 5])
prod5 = Product(5, 1.00, secondary_list = [1, 4])

products = [prod1, prod2, prod3, prod4, prod5]

""" Alpha functions """
mv = 1  # don't change it,  max alpha function value
act = 20  # activation
alpha_usr1 = [
    util.new_alpha_function(saturation_speed = 0.025, max_value = mv, activation = act),
    util.new_alpha_function(saturation_speed = 0.06, max_value = mv, activation = act),
    util.new_alpha_function(saturation_speed = 0.03, max_value = mv, activation = act),
    util.new_alpha_function(saturation_speed = 0.04, max_value = mv, activation = act),
    util.new_alpha_function(saturation_speed = 0.04, max_value = mv, activation = act),
    util.new_alpha_function(saturation_speed = 0.06, max_value = mv, activation = act)
]
act = 25  # activation
alpha_usr2 = [
    util.new_alpha_function(saturation_speed = 0.18, max_value = mv, activation = act),
    util.new_alpha_function(saturation_speed = 0.16, max_value = mv, activation = act),
    util.new_alpha_function(saturation_speed = 0.23, max_value = mv, activation = act),
    util.new_alpha_function(saturation_speed = 0.25, max_value = mv, activation = act),
    util.new_alpha_function(saturation_speed = 0.21, max_value = mv, activation = act),
    util.new_alpha_function(saturation_speed = 0.19, max_value = mv, activation = act)
]
act = 15  # activation
alpha_usr3 = [
    util.new_alpha_function(saturation_speed = 0.08, max_value = mv, activation = act),
    util.new_alpha_function(saturation_speed = 0.12, max_value = mv, activation = act),
    util.new_alpha_function(saturation_speed = 0.07, max_value = mv, activation = act),
    util.new_alpha_function(saturation_speed = 0.10, max_value = mv, activation = act),
    util.new_alpha_function(saturation_speed = 0.07, max_value = mv, activation = act),
    util.new_alpha_function(saturation_speed = 0.09, max_value = mv, activation = act)
]

# plot alpha functions
test_alpha = alpha_usr3
img, axss = plt.subplots(nrows = 2, ncols = 3, figsize = (13, 6))
axs = axss.flatten()
for i in range(6):
    x = np.linspace(0, 100, 100)
    y = test_alpha[i](x).clip(0.0)  # visual clip, be careful using them plain
    axs[i].set_xlabel("budget")
    axs[i].set_ylabel(f"alpha{i}_val")
    axs[i].plot(x, y)

if False:  # show or not alpha plot
    plt.show()

""" Users SETUP """

prob_user1 = 0.25
prob_user2 = 0.45
prob_user3 = 0.30

prob_users = [prob_user1, prob_user2, prob_user3]

# base         [0.50, 0.625, 0.75, 0.875, 1.00]
res_prices_1 = [0.60, 0.725, 0.65, 0.975, 1.10]
res_prices_2 = [0.60, 0.525, 0.85, 0.775, 1.10]
res_prices_3 = [0.60, 0.725, 0.85, 0.975, 1.10]

exp_number_purchase_1 = [2.0, 2.0, 2.0, 2.0, 3.5]  # 2   - 3.5
exp_number_purchase_2 = [1.0, 1.0, 1.0, 1.0, 1.5]  # 1   - 1.5
exp_number_purchase_3 = [1.5, 1.5, 1.5, 1.5, 2.0]  # 1.5 - 2


user1 = User(id = 1,
             reservation_prices = res_prices_1,
             lmbda = 0.8,
             alpha_functions = alpha_usr1,
             exp_number_purchase = exp_number_purchase_1,
             weighted_graph=util.random_fully_connected_graph(products)
             )

user2 = User(id = 2,
             reservation_prices = res_prices_2,
             lmbda = 0.5,
             alpha_functions = alpha_usr2,
             exp_number_purchase = exp_number_purchase_2,
             weighted_graph=util.random_fully_connected_graph(products)
             )

user3 = User(id = 3,
             reservation_prices = res_prices_3,
             lmbda = 0.65,
             alpha_functions = alpha_usr3,
             exp_number_purchase = exp_number_purchase_3,
             weighted_graph=util.random_fully_connected_graph(products),
             )

users = [user1, user2, user3]

""" Campaigns SETUP """
alpha_i_max = [0.5, 0.5, 0.3, 0.4, 0.3]
allocated_budget = [10, 20, 30, 40, 50,60,70,80,90,100]  # range 0-100
competitor_alpha = np.sum(alpha_i_max)

cmp1 = Campaign(1, allocated_budget[0], alpha_i_max = alpha_i_max[0])
cmp2 = Campaign(2, allocated_budget[1], alpha_i_max = alpha_i_max[1])
cmp3 = Campaign(3, allocated_budget[2], alpha_i_max = alpha_i_max[2])
cmp4 = Campaign(4, allocated_budget[3], alpha_i_max = alpha_i_max[3])
cmp5 = Campaign(5, allocated_budget[4], alpha_i_max = alpha_i_max[4])

campaigns = [cmp1, cmp2, cmp3, cmp4, cmp5]

""" Multiple days simulation """

days = 21
N_user = 1000  # reference for what alpha= 1 refers to
reference_price = 3.5

for day in range(days):
    cmp1.change_budget(5 * day)
    cmp2.change_budget(5 * day)
    cmp3.change_budget(5 * day)
    cmp4.change_budget(5 * day)
    cmp5.change_budget(5 * day)

    gross_profit = 0
    """ test total profit by all campaign and all users"""
    for i, cmp in enumerate(campaigns):
        # can be read: expected profit with respect the total number of users and the reference price
        gross_profit += (prob_user1 * cmp.get_alpha_i(user1.alpha_functions[i]) * user1.expected_profit()[i] +
                         prob_user2 * cmp.get_alpha_i(user2.alpha_functions[i]) * user2.expected_profit()[i] +
                         prob_user3 * cmp.get_alpha_i(user3.alpha_functions[i]) * user3.expected_profit()[i])

    # convert the pure number in euro
    gross_profit_euro = gross_profit * N_user * reference_price

    print(f"invested: {5 * day * 5:.2f}€ daily profit: {gross_profit_euro:.2f}€")

    """# behavior of USERS based on their alpha_function for product 1 and campaign 1 (aggregate alpha_1)
    alpha_1 = (prob_user1 * cmp1.get_alpha_i(user1.alpha_functions[1]) +
               prob_user2 * cmp1.get_alpha_i(user2.alpha_functions[1]) +
               prob_user3 * cmp1.get_alpha_i(user3.alpha_functions[1]))
    # behavior of USER1 based on their alpha_function for product 1 and campaign 1 (disaggregated alpha_1)
    alpha_1_usr1 = prob_user1 * cmp1.get_alpha_i(user1.alpha_functions[1])

    # expected gross profit of product one (aggregated profit)
    gross_profit_prod1 = (prob_user1 * cmp1.get_alpha_i(user1.alpha_functions[1]) * user1.expected_profit()[0] +
                          prob_user2 * cmp1.get_alpha_i(user2.alpha_functions[1]) * user2.expected_profit()[0] +
                          prob_user3 * cmp1.get_alpha_i(user3.alpha_functions[1]) * user3.expected_profit()[0])
    # expected gross profit of product one user one (disaggregated profit)
    gross_profit_prod1_usr1 = (prob_user1 * cmp1.get_alpha_i(user1.alpha_functions[1]) * user1.expected_profit()[0])
    gross_profit_prod1_usr2 = (prob_user2 * cmp1.get_alpha_i(user2.alpha_functions[1]) * user2.expected_profit()[0])
    gross_profit_prod1_usr3 = (prob_user3 * cmp1.get_alpha_i(user3.alpha_functions[1]) * user3.expected_profit()[0])

    print(
        f"with {allocated_budget} budget -> alpha1: {alpha_1:.3f}/{alpha_i_max[0]}\t Profit {gross_profit_prod1:.3f}"
        f"\t(u1: {gross_profit_prod1_usr1:.3f}) (u2: {gross_profit_prod1_usr2:.3f}) (u3: {gross_profit_prod1_usr3:.3f})")"""

""" Knapsack optimization simulation """

""" Some constants """

N_CLASSES = len(users)
N_CAMPAIGNS = len(products)
N_BUDGETS = len(allocated_budget)

print("*" * 30 + " Knapsack simulation " + "*" * 35)
print()

rewards = np.zeros((N_CLASSES * N_CAMPAIGNS, N_BUDGETS), dtype = np.single)

for cmp_index in range(0, N_CAMPAIGNS):

    for budget_idx in range(0, N_BUDGETS):

        """ Each campaign is observed with all the budget values """
        cmp = Campaign(id = cmp_index + 1,
                       allocated_budget = allocated_budget[budget_idx],
                       alpha_i_max = alpha_i_max[cmp_index])

        for user_idx, user in enumerate(users):
            prob_user = prob_users[user_idx]
            alpha = cmp.get_alpha_i(user.alpha_functions[cmp_index])
            value_per_click = user.expected_profit()[cmp_index]
            expected_gross_profit = prob_user * alpha * value_per_click
            rewards[cmp_index * N_CLASSES + user_idx][budget_idx] = np.single(expected_gross_profit)


col_labels = [str(budget) for budget in allocated_budget]

rewards = reference_price * N_user * rewards  # convert the pure number rewards in euros

print("*" * 10 + " Independent rewards Table " + "*" * 10)


row_label_rewards = []
row_labels_dp_table =['0']
for i in range(1, N_CAMPAIGNS+1):
    for j in range(1, N_CLASSES+1):

        # Cij -> campaign i and user j
        row_label_rewards.append("C" + str(i) + str(j))
        row_labels_dp_table.append("+C" + str(i) + str(j))


print(pd.DataFrame(rewards, columns = col_labels, index = row_label_rewards))

K = Knapsack(rewards = rewards, budgets = np.array(allocated_budget))

K.init_for_pretty_print(row_labels = row_labels_dp_table, col_labels = col_labels)
K.solve()

print()

K.pretty_print_dp_table()  # prints the final dynamic programming table

print()

K.pretty_print_output(
    print_last_row_only = False)  # prints information about last row of the table, including allocations


