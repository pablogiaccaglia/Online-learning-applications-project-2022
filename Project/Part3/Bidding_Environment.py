from User import User
from Campaign import Campaign
from Product import Product
import numpy as np


class Bidding_Environment:
    def __init__(self, users, campaigns, prob_users):
        self.users = users
        self.campaigns = campaigns
        self.n_campaigns = len(campaigns)
        self.prob_users = prob_users

    def get_aggregated_profit(self, allocated_budgets):
        rewards_per_campaign = []
        for cmp_index in range(0, self.n_campaigns):

            """ Each campaign is observed with all the budget values """
            cmp = self.campaigns[cmp_index]
            cmp.change_budget(allocated_budgets[cmp_index])
            expected_gross_profit = 0.0
            for user_idx, user in enumerate(self.users):
                prob_user = self.prob_users[user_idx]
                alpha = cmp.get_alpha_i(user.alpha_functions[cmp_index])
                value_per_click = user.expected_profit(np.ones(5))[cmp_index]
                expected_gross_profit += prob_user * alpha * value_per_click
            rewards_per_campaign.append(expected_gross_profit)
        return rewards_per_campaign
