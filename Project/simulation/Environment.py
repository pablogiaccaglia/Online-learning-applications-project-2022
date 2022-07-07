import numpy as np
import matplotlib.pyplot as plt
import Utils as util
from Campaign import Campaign
from Product import Product
from User import User


class Environment:
    def __init__(self):
        # print("init env")
        """ Products SETUP """
        prod1 = Product(1, 0.50, secondary_list=[2, 3])
        prod2 = Product(2, 0.625, secondary_list=[3, 4])
        prod3 = Product(3, 0.75, secondary_list=[1, 5])
        prod4 = Product(4, 0.875, secondary_list=[2, 5])
        prod5 = Product(5, 1.00, secondary_list=[1, 4])
        self.products = [prod1, prod2, prod3, prod4, prod5]
        """ Alpha functions SETUP """
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

        # plot alpha functions
        do_plot = False
        test_alpha = alpha_usr3
        img, axss = plt.subplots(nrows=2, ncols=3, figsize=(13, 6))
        axs = axss.flatten()
        for i in range(6):
            x = np.linspace(0, 100, 100)
            y = test_alpha[i](x).clip(0.0)  # visual clip, be careful using them plain
            axs[i].set_xlabel("budget")
            axs[i].set_ylabel(f"alpha{i}_val")
            axs[i].plot(x, y)

        if do_plot:  # show or not alpha plot
            plt.show()

        """ Users SETUP """
        prob_user1 = 0.25
        prob_user2 = 0.45
        prob_user3 = 0.30

        # base         [0.50, 0.625, 0.75, 0.875, 1.00]
        res_prices_1 = [0.60, 0.725, 0.65, 0.975, 1.10]
        res_prices_2 = [0.60, 0.525, 0.85, 0.775, 1.10]
        res_prices_3 = [0.60, 0.725, 0.85, 0.975, 1.10]

        exp_number_purchase_1 = [2.0, 2.0, 2.0, 2.0, 3.5]  # 2   - 3.5
        exp_number_purchase_2 = [1.0, 1.0, 1.0, 1.0, 1.5]  # 1   - 1.5
        exp_number_purchase_3 = [1.5, 1.5, 1.5, 1.5, 2.0]  # 1.5 - 2

        graph1 = util.random_fully_connected_graph(self.products)
        graph2 = util.random_fully_connected_graph(self.products)
        graph3 = util.random_fully_connected_graph(self.products)

        user1 = User(id=1,
                     reservation_prices=res_prices_1,
                     lmbda=0.8,
                     alpha_functions=alpha_usr1,
                     exp_number_purchase=exp_number_purchase_1,
                     weighted_graph=graph1,
                     )

        user2 = User(id=2,
                     reservation_prices=res_prices_2,
                     lmbda=0.5,
                     alpha_functions=alpha_usr2,
                     exp_number_purchase=exp_number_purchase_2,
                     weighted_graph=graph2,
                     )

        user3 = User(id=3,
                     reservation_prices=res_prices_3,
                     lmbda=0.65,
                     alpha_functions=alpha_usr3,
                     exp_number_purchase=exp_number_purchase_3,
                     weighted_graph=graph3,
                     )

        """ Campaigns SETUP """
        alpha_i_max = [0.5, 0.5, 0.3, 0.4, 0.3]
        self.allocated_budget = [10, 20, 30, 40, 50]  # range 0-100

        cmp1 = Campaign(1, self.allocated_budget[0], alpha_i_max=alpha_i_max[0])
        cmp2 = Campaign(2, self.allocated_budget[1], alpha_i_max=alpha_i_max[1])
        cmp3 = Campaign(3, self.allocated_budget[2], alpha_i_max=alpha_i_max[2])
        cmp4 = Campaign(4, self.allocated_budget[3], alpha_i_max=alpha_i_max[3])
        cmp5 = Campaign(5, self.allocated_budget[4], alpha_i_max=alpha_i_max[4])

        """ Extended campaign setup """
        # TODO develop ideal situation when n_user * n_product campaigns are available
        # edit: abort this passage since the disaggregated knspsack is performing worse

        self.campaigns = [cmp1, cmp2, cmp3, cmp4, cmp5]
        self.competitor_alpha = np.sum(alpha_i_max)
        self.users = [user1, user2, user3]
        self.prob_users = [prob_user1, prob_user2, prob_user3]
        self.noise_alpha = []
        self.exp_number_noise = []

    def play_one_day(self, n_users, reference_price, daily_budget, alpha_noise=False, n_noise=False):
        # generate noisy contractions matrix for alpha functions and exp number of purchase
        if alpha_noise:
            self.noise_alpha = util.noise_matrix_alpha()
        else:
            self.noise_alpha = util.no_noise_matrix()
        if n_noise:
            self.exp_number_noise = util.noise_matrix_alpha(max_reduction=0.25, max_global_influence=0)
        else:
            self.exp_number_noise = util.no_noise_matrix()

        noise_alpha = self.noise_alpha
        exp_number_noise = self.exp_number_noise

        p1 = self.prob_users[0]
        p2 = self.prob_users[1]
        p3 = self.prob_users[2]
        u1 = self.users[0]
        u2 = self.users[1]
        u3 = self.users[2]
        profit_u1 = 0
        profit_u2 = 0
        profit_u3 = 0
        step_k = 15  # set step size for knapsack
        n_budget_k = int(daily_budget / step_k)  # adapt columns number for knapsack

        for i, cmp in enumerate(self.campaigns):
            # can be read: expected profit with respect the total number of users and the reference price
            profit_u1 += p1 * noise_alpha[0][i] * cmp.get_alpha_i(u1.alpha_functions[i]) * \
                         u1.expected_profit(exp_number_noise[0])[i]
            profit_u2 += p2 * noise_alpha[1][i] * cmp.get_alpha_i(u2.alpha_functions[i]) * \
                         u2.expected_profit(exp_number_noise[0])[i]
            profit_u3 += p3 * noise_alpha[2][i] * cmp.get_alpha_i(u3.alpha_functions[i]) * \
                         u3.expected_profit(exp_number_noise[0])[i]

        # convert the pure number in euro
        profit_u1_euro = profit_u1 * n_users * reference_price
        profit_u2_euro = profit_u2 * n_users * reference_price
        profit_u3_euro = profit_u3 * n_users * reference_price
        daily_profit_euro = profit_u1_euro + profit_u2_euro + profit_u3_euro

        # knapsack disaggregated reward computation
        rewards_k, avail_budgets = self.__rewards_knapsack(n_users,
                                                           reference_price,
                                                           noise_alpha,
                                                           exp_number_noise=exp_number_noise,
                                                           step_size=step_k,
                                                           n_budgets=n_budget_k)
        # knapsack aggregated reward computation
        rewards_k_agg, avail_budgets_agg = self.__rewards_knapsack_aggregated(n_users,
                                                                              reference_price,
                                                                              noise_alpha,
                                                                              exp_number_noise=exp_number_noise,
                                                                              step_size=step_k,
                                                                              n_budgets=n_budget_k)

        return {
            "profit": (profit_u1_euro, profit_u2_euro, profit_u3_euro, daily_profit_euro),
            "noise": (noise_alpha, exp_number_noise),
            "reward_k": (rewards_k, avail_budgets),
            "reward_k_agg": (rewards_k_agg, avail_budgets_agg),
        }

    def replicate_last_day(self, n_users, reference_price, alpha_noise=False, n_noise=False):
        if alpha_noise:
            noise_alpha = self.noise_alpha
        else:
            noise_alpha = util.no_noise_matrix()
        if n_noise:
            exp_number_noise = self.exp_number_noise
        else:
            exp_number_noise = util.no_noise_matrix()

        p1 = self.prob_users[0]
        p2 = self.prob_users[1]
        p3 = self.prob_users[2]
        u1 = self.users[0]
        u2 = self.users[1]
        u3 = self.users[2]
        profit_u1 = 0
        profit_u2 = 0
        profit_u3 = 0

        for i, cmp in enumerate(self.campaigns):
            # can be read: expected profit with respect the total number of users and the reference price
            profit_u1 += p1 * noise_alpha[0][i] * cmp.get_alpha_i(u1.alpha_functions[i]) * u1.expected_profit()[i]
            profit_u2 += p2 * noise_alpha[1][i] * cmp.get_alpha_i(u2.alpha_functions[i]) * u2.expected_profit()[i]
            profit_u3 += p3 * noise_alpha[2][i] * cmp.get_alpha_i(u3.alpha_functions[i]) * u3.expected_profit()[i]

        # convert the pure number in euro
        profit_u1_euro = profit_u1 * n_users * reference_price
        profit_u2_euro = profit_u2 * n_users * reference_price
        profit_u3_euro = profit_u3 * n_users * reference_price
        daily_profit_euro = profit_u1_euro + profit_u2_euro + profit_u3_euro

        return {
            "profit": (profit_u1_euro, profit_u2_euro, profit_u3_euro, daily_profit_euro),
            "noise": (noise_alpha, exp_number_noise)
        }

    def get_core_entities(self):
        return self.users, self.products, self.campaigns, self.allocated_budget, self.prob_users

    def set_campaign_budget(self, index, budget):
        self.campaigns[index].change_budget(budget)
        self.allocated_budget[index] = budget

    def __rewards_knapsack(self, n_users, reference_price, noise_alpha, exp_number_noise, step_size=5, n_budgets=10):

        # print("*" * 25 + " Knapsack rewards " + "*" * 30)

        available_budget = [step_size * (i + 1) for i in range(n_budgets)]
        n_classes = len(self.users)
        n_campaigns = len(self.products)

        rewards = np.zeros((n_classes * n_campaigns, n_budgets), dtype=np.single)

        for cmp_index in range(n_campaigns):
            for budget_idx in range(n_budgets):
                self.campaigns[cmp_index].change_budget(available_budget[budget_idx])
                for user_idx, user in enumerate(self.users):
                    alpha = self.campaigns[cmp_index].get_alpha_i(user.alpha_functions[cmp_index]) * \
                            noise_alpha[user_idx][
                                cmp_index]
                    value_per_click = user.expected_profit(exp_number_noise[user_idx])[cmp_index]
                    expected_gross_profit = self.prob_users[
                                                user_idx] * alpha * value_per_click * n_users * reference_price
                    rewards[cmp_index * n_classes + user_idx][budget_idx] = np.single(expected_gross_profit)

        return rewards, available_budget

    def __rewards_knapsack_aggregated(self, n_users, reference_price, noise_alpha, exp_number_noise, step_size=5, n_budgets=10):

        # print("*" * 25 + " Knapsack rewards " + "*" * 30)

        available_budget = [step_size * (i + 1) for i in range(n_budgets)]
        n_campaigns = len(self.products)

        rewards = np.zeros((n_campaigns, n_budgets), dtype=np.single)

        for cmp_index in range(n_campaigns):
            for budget_idx in range(n_budgets):
                self.campaigns[cmp_index].change_budget(available_budget[budget_idx])
                for user_idx, user in enumerate(self.users):
                    alpha = self.campaigns[cmp_index].get_alpha_i(user.alpha_functions[cmp_index]) * \
                            noise_alpha[user_idx][
                                cmp_index]
                    value_per_click = user.expected_profit(exp_number_noise[user_idx])[cmp_index]
                    expected_gross_profit = self.prob_users[
                                                user_idx] * alpha * value_per_click * n_users * reference_price
                    rewards[cmp_index][budget_idx] += np.single(expected_gross_profit)

        return rewards, available_budget
