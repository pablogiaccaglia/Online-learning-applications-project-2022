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
        act = 23  # activation
        alpha_usr1 = [
            util.new_alpha_function(saturation_speed=0.025, max_value=mv, activation=act),
            util.new_alpha_function(saturation_speed=0.06, max_value=mv, activation=act),
            util.new_alpha_function(saturation_speed=0.03, max_value=mv, activation=act),
            util.new_alpha_function(saturation_speed=0.04, max_value=mv, activation=act),
            util.new_alpha_function(saturation_speed=0.04, max_value=mv, activation=act),
            util.new_alpha_function(saturation_speed=0.06, max_value=mv, activation=act)
        ]
        act = 30  # activation
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
        if do_plot:  # show or not alpha plot
            test_alpha = alpha_usr3
            img, axss = plt.subplots(nrows=2, ncols=3, figsize=(13, 6))
            axs = axss.flatten()
            for i in range(6):
                x = np.linspace(0, 100, 100)
                y = test_alpha[i](x).clip(0.0)  # visual clip, be careful using them plain
                axs[i].set_xlabel("budget")
                axs[i].set_ylabel(f"alpha{i}_val")
                axs[i].plot(x, y)
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
        alpha_i_max = [0.4, 0.4, 0.2, 0.3, 0.2]
        self.allocated_budget = [40, 40, 40, 40, 40]  # range 0-100

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

    def play_one_day(self, n_users, reference_price, daily_budget, step_k=2, alpha_noise=False, n_noise=False):
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

        n_budget_k = int(daily_budget / step_k)  # adapt columns number for knapsack

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
            "profit": self.__profit_per_user(n_users, reference_price),
            "noise": (noise_alpha, exp_number_noise),
            "reward_k": (rewards_k, avail_budgets),
            "reward_k_agg": (rewards_k_agg, avail_budgets_agg),
            "profit_campaign": self.__profit_per_campaign(n_users, reference_price),
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

        return {
            "profit": self.__profit_per_user(n_users, reference_price),
            "noise": (noise_alpha, exp_number_noise),
            "profit_campaign": self.__profit_per_campaign(n_users, reference_price),
        }

    def get_core_entities(self):
        return self.users, self.products, self.campaigns, self.allocated_budget, self.prob_users

    def set_campaign_budget(self, index, budget):
        self.campaigns[index].change_budget(budget)
        self.allocated_budget[index] = budget

    def get_context_building_blocks(self, budgets_array, n_users, reference_price):
        """ return knapsack rewards and profits for each possible split of the context"""
        # noise replication as last available data in env
        if len(budgets_array) != 4:
            raise ValueError("Illegal budget array size")
        old_budget = self.allocated_budget
        blocks_p = []
        # trick for user1 appearing 2 times
        indexes = [0, 0, 1, 2]

        for budget_i, user_i in enumerate(indexes):
            budget = budgets_array[budget_i]
            if len(budget) != len(self.campaigns):
                raise ValueError(f"Illegal {budget_i} budget size")
            # force a temporary budget
            self.__set_campaign_budgets(budget)

            profit = np.array(self.__profit_campaign_per_user(user_i, n_users, reference_price))
            blocks_p.append(profit)

        self.__set_campaign_budgets(old_budget)  # restore old budget
        return blocks_p

    def __set_campaign_budgets(self, budget_array):
        for i, cmp in enumerate(self.campaigns):
            self.campaigns[i].change_budget(budget_array[i])
            self.allocated_budget[i] = budget_array[i]

    def __rewards_knapsack(self, n_users, reference_price, noise_alpha, exp_number_noise, step_size=5, n_budgets=10):

        # print("*" * 25 + " Knapsack rewards " + "*" * 30)
        old_budget = self.allocated_budget

        available_budget = [step_size * (i + 1) for i in range(n_budgets)]
        n_classes = len(self.users)
        n_campaigns = len(self.products)

        rewards = []
        for _ in range(n_classes * n_campaigns):
            rewards.append(-1 * np.array(available_budget.copy()))
        rewards = np.array(rewards)

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
                    rewards[cmp_index * n_classes + user_idx][budget_idx] += np.single(expected_gross_profit)

        self.__set_campaign_budgets(old_budget)  # restore old budget
        return rewards, available_budget

    def __rewards_knapsack_aggregated(self, n_users, reference_price, noise_alpha, exp_number_noise, step_size=5,
                                      n_budgets=10):

        # print("*" * 25 + " Knapsack rewards " + "*" * 30)
        old_budget = self.allocated_budget

        available_budget = [step_size * (i + 1) for i in range(n_budgets)]
        n_campaigns = len(self.products)

        rewards = []
        for _ in range(n_campaigns):
            rewards.append(-1 * np.array(available_budget.copy()))
        rewards = np.array(rewards)

        for cmp_index in range(n_campaigns):
            for budget_idx in range(n_budgets):
                for user_idx, user in enumerate(self.users):
                    self.campaigns[cmp_index].change_budget(available_budget[budget_idx] * self.prob_users[
                        user_idx])  # scale allocated budget by probability of user
                    alpha = self.campaigns[cmp_index].get_alpha_i(user.alpha_functions[cmp_index]) * \
                            noise_alpha[user_idx][
                                cmp_index]
                    value_per_click = user.expected_profit(exp_number_noise[user_idx])[cmp_index]
                    expected_gross_profit = self.prob_users[
                                                user_idx] * alpha * value_per_click * n_users * reference_price
                    rewards[cmp_index][budget_idx] += np.single(expected_gross_profit)

        self.__set_campaign_budgets(old_budget)  # restore old budget
        return rewards, available_budget

    def __profit_per_user(self, n_users, reference_price):
        """ Alert -> It scales the budget by user probability: use it for aggregated case """
        noise_alpha = self.noise_alpha
        exp_number_noise = self.exp_number_noise

        old_budget = self.allocated_budget  # save old budget
        profit_u = [0, 0, 0]

        for i, cmp in enumerate(self.campaigns):
            for j, u in enumerate(self.users):
                # cumulative expected profit for user j over all campaigns
                p_u = self.prob_users[j]  # user j probability
                self.campaigns[i].change_budget(old_budget[i] * p_u)  # allocated scaled budget by user probability
                noise_a = noise_alpha[j][i]  # noise over (i,j)
                alpha_f_res = cmp.get_alpha_i(u.alpha_functions[i])  # effect of budget over campaign (alpha function)
                exp_profit = u.expected_profit(exp_number_noise[j])[i]  # expected profit of user j over graph + noise n effect

                profit_u[j] += p_u * noise_a * alpha_f_res * exp_profit
                # todo TEST !!

        # convert the pure number in euro
        profit_u1_euro = profit_u[0] * n_users * reference_price
        profit_u2_euro = profit_u[1] * n_users * reference_price
        profit_u3_euro = profit_u[2] * n_users * reference_price
        daily_profit_euro = profit_u1_euro + profit_u2_euro + profit_u3_euro

        self.__set_campaign_budgets(old_budget)  # restore old budget

        return profit_u1_euro, profit_u2_euro, profit_u3_euro, daily_profit_euro

    def __profit_per_campaign(self, n_users, reference_price):
        """ Alert -> It scales the budget by user probability: use it for aggregated case """
        noise_alpha = self.noise_alpha
        exp_number_noise = self.exp_number_noise

        old_budget = self.allocated_budget  # save old budget
        campaign_profits = []

        for i, cmp in enumerate(self.campaigns):
            tmp_profit = 0
            for j, u in enumerate(self.users):
                # cumulative expected profit for user j over all campaigns
                p_u = self.prob_users[j]  # user j probability
                self.campaigns[i].change_budget(old_budget[i] * p_u)  # allocated scaled budget by user probability
                noise_a = noise_alpha[j][i]  # noise over (i,j)
                alpha_f_res = cmp.get_alpha_i(u.alpha_functions[i])  # effect of budget over campaign (alpha function)
                exp_profit = u.expected_profit(exp_number_noise[j])[i]  # expected profit of user j over graph + noise n effect

                tmp_profit += p_u * noise_a * alpha_f_res * exp_profit
            # todo TEST !!
            campaign_profits.append(tmp_profit)

        # convert the pure number in euro
        for i, cmp_profit in enumerate(campaign_profits):
            euro_val = cmp_profit * n_users * reference_price
            campaign_profits[i] = euro_val

        campaign_profits.append(np.sum(campaign_profits))
        self.__set_campaign_budgets(old_budget)  # restore old budget

        return tuple(campaign_profits)

    def __profit_campaign_per_user(self, user_index, n_users, reference_price):
        """ Not scaled user probability profit for a campaign with forced budget of caller"""
        noise_alpha = self.noise_alpha
        exp_number_noise = self.exp_number_noise
        campaign_profits = []
        u = self.users[user_index]

        for i, cmp in enumerate(self.campaigns):
            # can be read: expected profit with respect the total number of users and the reference price
            profit_u = noise_alpha[user_index][i] * cmp.get_alpha_i(u.alpha_functions[i]) * \
                       u.expected_profit(exp_number_noise[user_index])[i]
            campaign_profits.append(profit_u)

        # convert the pure number in euro
        for i, cmp_profit in enumerate(campaign_profits):
            euro_val = cmp_profit * n_users * reference_price
            campaign_profits[i] = euro_val

        campaign_profits.append(np.sum(campaign_profits))

        return tuple(campaign_profits)
