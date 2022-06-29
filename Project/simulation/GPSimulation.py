import numpy as np
import matplotlib.pyplot as plt
import Project.Utils as util
from Project.Campaign import Campaign
from Project.Product import Product
from Project.User import User

""" How to create alpha functions """
test_alpha = [
    util.new_alpha_function(saturation_speed=0.08, max_value=0.9, activation=20),
    util.new_alpha_function(saturation_speed=0.1, max_value=0.85, activation=30),
    util.new_alpha_function(saturation_speed=0.3, max_value=0.85, activation=25),
    util.new_alpha_function(saturation_speed=0.14, max_value=0.7, activation=25),
    util.new_alpha_function(saturation_speed=0.11, max_value=0.9, activation=20),
    util.new_alpha_function(saturation_speed=0.09, max_value=0.8, activation=15)
]

""" Create 5 Products """

prod1 = Product(1, 2.0, secondary_list=[2, 3])
prod2 = Product(2, 2.5, secondary_list=[3, 4])
prod3 = Product(3, 3.0, secondary_list=[1, 5])
prod4 = Product(4, 4.0, secondary_list=[2, 5])
prod5 = Product(5, 4.5, secondary_list=[1, 4])

products = [prod1, prod2, prod3, prod4, prod5]


""" Create 3 users """

user1 = User(id=1,
             reservation_prices=[6.0, 6.0, 6.0, 6.0, 6.0],
             lmbda=0.7,
             alpha_functions=test_alpha[:5],
             exp_number_purchase=[2.1, 2.0, 1.0, 1.0, 0.0],
             weighted_graph=util.random_fully_connected_graph(products),
             # clarify: is the weighted graph per user or general?
             # expected_profits = expected_profits
             )

user2 = User(id=2,
             reservation_prices=[1.0, 2.0, 3.0, 4.0, 5.0],
             lmbda=0.85,
             alpha_functions=test_alpha[:5],
             exp_number_purchase=[1.4, 2.2, 1.3, 1.0, 1.1],
             weighted_graph=util.random_fully_connected_graph(products),
             # expected_profits = expected_profits
             )

user3 = User(id=3,
             reservation_prices=[6.0, 3.0, 1.0, 1.5, 1.3],
             lmbda=0.6,
             alpha_functions=test_alpha[:5],
             exp_number_purchase=[3.0, 2.0, 2.1, 1.6, 1.1],
             weighted_graph=util.random_fully_connected_graph(products),
             # expected_profits = expected_profits
             )

users = [user1, user2, user3]

""" Proof of concept on how to simulate multiple days """

# the sum of alpha_max is the maximum number of people that can be reached
alpha_i_max = [0.6, 0.5, 0.6, 0.6, 0.6]
competitor_alpha = np.sum(alpha_i_max)

days = 20

for day in range(days):
    # setup the campaign
    allocated_budget = 20 + 2 * day
    cmp1 = Campaign(1, allocated_budget, alpha_i_max=alpha_i_max[0])
    """cmp2 = Campaign(2, 10, alpha_i_max=alpha_i_max[1])
    cmp3 = Campaign(3, 10, alpha_i_max=alpha_i_max[2])
    cmp4 = Campaign(4, 10, alpha_i_max=alpha_i_max[3])
    cmp5 = Campaign(5, 10, alpha_i_max=alpha_i_max[4])"""

    campaigns = [cmp1]  # only 1 campaign for simplicity

    # define user probabilities
    prob_user1 = 0.2
    prob_user2 = 0.5
    prob_user3 = 0.3


    # expected gross profit of product one (aggregated profit)
    gross_profit_prod1 = (prob_user1 * cmp1.get_alpha_i(user1.alpha_functions[1]) * user1.expected_profit()[0] +
                          prob_user2 * cmp1.get_alpha_i(user2.alpha_functions[1]) * user2.expected_profit()[0] +
                          prob_user3 * cmp1.get_alpha_i(user3.alpha_functions[1]) * user3.expected_profit()[0])
    # expected gross profit of product one user one (disaggregated profit)
    gross_profit_prod1_usr1 = (cmp1.get_alpha_i(user1.alpha_functions[1]) * user1.expected_profit()[0])
    gross_profit_prod1_usr2 = (cmp1.get_alpha_i(user2.alpha_functions[1]) * user2.expected_profit()[0])
    gross_profit_prod1_usr3 = (cmp1.get_alpha_i(user3.alpha_functions[1]) * user3.expected_profit()[0])

    print()
    print (gross_profit_prod1_usr1/user1.expected_profit()[0])
