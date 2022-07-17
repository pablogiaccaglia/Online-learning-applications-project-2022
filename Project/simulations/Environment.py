import numpy as np
import matplotlib.pyplot as plt

from entities.Campaign import Campaign
from entities.Product import Product
import entities.Utils as util
from entities.User import User
from knapsack.Knapsack import Knapsack
from learners.OfflineWeightsLearner import OfflineWeightsLearner
from learners.OnlineWeightsLearner import OnlineWeightsLearner


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

        exp_number_purchase_1 = [2.5, 3, 2.0, 2.5, 2.0]  # 2   - 3.5
        exp_number_purchase_2 = [1.5, 1.5, 1.1, 1.2, 1.3]  # 1   - 1.5
        exp_number_purchase_3 = [2, 1.6, 1.8, 2.0, 1.5]  # 1.5 - 2

        graph1 = util.random_fully_connected_graph(self.products)
        graph2 = util.random_fully_connected_graph(self.products)
        graph3 = util.random_fully_connected_graph(self.products)

        self.graphs = [graph1, graph2, graph3]

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
        self.campaigns = [cmp1, cmp2, cmp3, cmp4, cmp5]
        self.competitor_alpha = np.sum(alpha_i_max)
        self.users = [user1, user2, user3]
        self.prob_users = [prob_user1, prob_user2, prob_user3]
        self.all_prob_users = [prob_user1 * 0.5, prob_user1 * 0.5, prob_user2, prob_user3]
        self.noise_alpha = []
        self.exp_number_noise = []

    def play_one_day(self, n_users, reference_price, daily_budget, step_k=2, alpha_noise=False, n_noise=False,
                     contexts=None):
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

        agg_rewards, avail_budgets = self.__rewards_knapsack_aggregated(n_users,
                                                                        reference_price,
                                                                        noise_alpha,
                                                                        exp_number_noise=exp_number_noise,
                                                                        step_size=step_k,
                                                                        n_budgets=n_budget_k)

        # knapsack disaggregated for 4 users
        rewards, _ = self.__rewards_knapsack_4_user(n_users,
                                                    reference_price,
                                                    noise_alpha,
                                                    exp_number_noise=exp_number_noise,
                                                    step_size=step_k,
                                                    n_budgets=n_budget_k)
        rewards2 = []
        if contexts is not None and len(contexts) == 2:
            for mask in contexts:
                tmp_r, b = self.__rewards_knapsack_pseudo_aggregated(mask,
                                                                     n_users,
                                                                     reference_price,
                                                                     noise_alpha,
                                                                     exp_number_noise=exp_number_noise,
                                                                     step_size=step_k,
                                                                     n_budgets=n_budget_k)
                rewards2.append(tmp_r)
            size = len(rewards2[0][0])
            rewards2 = np.array(rewards2).reshape((-1, size))

        # AGGREGATED
        row_label_rewards, row_labels_dp_table, col_labels = util.table_metadata(5, 1, avail_budgets)
        K = Knapsack(rewards=agg_rewards, budgets=np.array(avail_budgets))
        K.init_for_pretty_print(row_labels=row_labels_dp_table, col_labels=col_labels)
        K.solve()
        arg_max = np.argmax(K.get_output()[0][-1])
        alloc_agg = np.array(K.get_output()[1][-1][arg_max])[1:]
        # b_knap = budget_array_from_k_alloc_4(alloc)  # budgets vector for contextualized env
        reward = K.get_output()[0][-1][arg_max]
        reward_k_agg = reward

        # DISAGGREGATE
        row_label_rewards, row_labels_dp_table, col_labels = util.table_metadata(5, 4, avail_budgets)
        K = Knapsack(rewards=rewards, budgets=np.array(avail_budgets))
        K.init_for_pretty_print(row_labels=row_labels_dp_table, col_labels=col_labels)
        K.solve()
        arg_max = np.argmax(K.get_output()[0][-1])
        alloc_disagg = np.array(K.get_output()[1][-1][arg_max])[1:]
        # b_knap = budget_array_from_k_alloc_4(alloc)  # budgets vector for contextualized env
        reward = K.get_output()[0][-1][arg_max]
        reward_k_disagg = reward

        return {
            "k_budgets": avail_budgets,
            "noise": (noise_alpha, exp_number_noise),
            "reward_k_disagg": reward_k_disagg,
            "rewards_agg": agg_rewards,
            "rewards_disagg": rewards,
            "rewards_mix": rewards2,
            "reward_k_agg": reward_k_agg,
            "alloc_agg": (alloc_agg, np.sum(alloc_agg)),
            "alloc_disagg": (alloc_disagg, np.sum(alloc_disagg))
        }

    def replicate_last_day(self, super_arm, n_users, reference_price, alpha_noise=False, n_noise=False, contexts=None):
        if alpha_noise:
            noise_alpha = self.noise_alpha
        else:
            noise_alpha = util.no_noise_matrix()
        if n_noise:
            exp_number_noise = self.exp_number_noise
        else:
            exp_number_noise = util.no_noise_matrix()

        if contexts is None:
            ctx = [[1, 1, 1, 1]]
        else:
            ctx = contexts
        # adapt budget from super arm to the 4 class of users
        # OK TESTED
        budgets_array = self.budget_array_from_superarm(super_arm, ctx)
        # generate 4 profit blocks one per user
        # OK TESTED
        profit_blocks = self.get_context_building_blocks(budgets_array=budgets_array,
                                                         n_users=n_users,
                                                         reference_price=reference_price)
        # aggregate them according to the given context
        # OK TESTED
        gross_rewards = self.assemble_profit(profit_blocks, ctx, flatten=True)
        gross_rewards2 = self.assemble_profit(profit_blocks, ctx)
        learner_rewards = gross_rewards - np.array(super_arm)

        return {
            "learner_rewards": learner_rewards,
            "gross_rewards": gross_rewards,
            "noise": (noise_alpha, exp_number_noise),
            "profit": np.sum(learner_rewards),
        }

    def get_core_entities(self):
        return self.users, self.products, self.campaigns, self.allocated_budget, self.prob_users, self.graphs

    def set_campaign_budget(self, index, budget):
        self.campaigns[index].change_budget(budget)
        self.allocated_budget[index] = budget

    def get_context_building_blocks(self, budgets_array, n_users, reference_price):
        """ return knapsack rewards and profits for each possible split of the context
            Alert -> The result is not scaled by user probability"""
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

    def budget_array_from_superarm(self, super_arm, contexts):
        """ map the super arm result with blocks of budget for every possible context participant """
        p_users = self.all_prob_users
        if len(super_arm) / len(contexts) != 5.0:
            raise ValueError(f"Super arm not compatible with context {len(super_arm)}/{len(contexts)} != 5 \n "
                             f"{super_arm} || {contexts}")
        budgets = np.array(super_arm).reshape((len(contexts), 5))
        result = np.zeros((4, 5))
        for i, ctx in enumerate(contexts):
            mask = np.array(ctx)
            scaled_mask = mask * np.array(p_users) / np.sum(mask * np.array(p_users))
            scaled_mask = scaled_mask.reshape((len(ctx), 1))
            b = budgets[i].reshape((1, 5))
            result += scaled_mask @ b

        return result  # matrix of scaled budgets

    def __set_campaign_budgets(self, budget_array):
        for i, cmp in enumerate(self.campaigns):
            self.campaigns[i].change_budget(budget_array[i])
            self.allocated_budget[i] = budget_array[i]

    def assemble_profit(self, profit_blocks, contexts, flatten=False):
        """Perform addition and scale by user probability in the context"""
        assembled_profit = []
        p_users = self.all_prob_users
        block = np.array(profit_blocks).T  # transpose profit to (camp x usr)
        for mask in contexts:
            scaled_mask = np.array(mask) * np.array(p_users)
            context_profit = block @ scaled_mask
            assembled_profit.append(context_profit)
        if flatten:
            return np.array(assembled_profit).flatten()

        return np.array(assembled_profit)

    def budget_array_from_k_alloc_4(self, _alloc, flatten=False):
        """ map the knapsack allocation result with blocks of budget for every possible context participant """
        # _alloc = [0, 1, 2, 3, 11, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 333]
        if len(_alloc) != 21:
            raise ValueError("Knapsack disaggregated alloc needed")
        alloc_clean = _alloc[1:]  # remove 0
        a = np.array(alloc_clean).reshape((5, -1))  # reshape in cluster of 4 res x 5 camp
        tmp = np.reshape(a, len(alloc_clean), order='F')  # rorder per user
        tmp = tmp.reshape((-1, 5))  # reshape 5 camp x 4 user

        if flatten:
            return tmp.flatten()
        return tmp

    def __rewards_knapsack_4_user(self, n_users, reference_price, noise_alpha, exp_number_noise, step_size=5,
                                  n_budgets=10):
        """Return knapsack rewards compatible with full split of user classes"""
        old_budget = self.allocated_budget

        available_budget = [step_size * (i + 1) for i in range(n_budgets)]
        n_classes = 4
        n_campaigns = len(self.products)
        prob_users = self.all_prob_users
        users = [self.users[0], self.users[0], self.users[1], self.users[2]]

        rewards = []
        for _ in range(n_classes * n_campaigns):
            rewards.append(-1 * np.array(available_budget.copy()))
        rewards = np.array(rewards)

        for cmp_index, cmp in enumerate(self.campaigns):
            for budget_idx in range(n_budgets):
                cmp.change_budget(available_budget[budget_idx])
                for user_idx, user in enumerate(users):
                    true_idx = user_idx
                    if user_idx >= 1:
                        true_idx = user_idx - 1  # normalization to not replicate alpha noise
                    alpha = cmp.get_alpha_i(user.alpha_functions[cmp_index]) * \
                            noise_alpha[true_idx][
                                cmp_index]
                    value_per_click = user.expected_profit(exp_number_noise[true_idx])[cmp_index]
                    expected_gross_profit = prob_users[
                                                user_idx] * alpha * value_per_click * n_users * reference_price
                    rewards[cmp_index * n_classes + user_idx][budget_idx] += np.single(expected_gross_profit)

        self.__set_campaign_budgets(old_budget)  # restore old budget
        return rewards, available_budget

    def __rewards_knapsack_aggregated(self, n_users, reference_price, noise_alpha, exp_number_noise, step_size=5,
                                      n_budgets=10, mask=None):
        """Return knapsack rewards for fully aggregated user classes"""
        old_budget = self.allocated_budget
        prob_users = self.all_prob_users[1:]
        available_budget = [step_size * (i + 1) for i in range(n_budgets)]
        n_campaigns = len(self.products)
        rewards = []
        for _ in range(n_campaigns):
            rewards.append(-1 * np.array(available_budget.copy()))
        rewards = np.array(rewards)

        for cmp_index in range(n_campaigns):
            for budget_idx in range(n_budgets):
                for user_idx, user in enumerate(self.users):
                    self.campaigns[cmp_index].change_budget(available_budget[budget_idx] * prob_users[
                        user_idx])  # scale allocated budget by probability of user
                    alpha = self.campaigns[cmp_index].get_alpha_i(user.alpha_functions[cmp_index]) * \
                            noise_alpha[user_idx][cmp_index]
                    value_per_click = user.expected_profit(exp_number_noise[user_idx])[cmp_index]
                    expected_gross_profit = prob_users[
                                                user_idx] * alpha * value_per_click * n_users * reference_price
                    multiplier = 2 if user_idx == 0 else 1
                    rewards[cmp_index][budget_idx] += np.single(expected_gross_profit) * multiplier

        self.__set_campaign_budgets(old_budget)  # restore old budget
        return rewards, available_budget

    def __rewards_knapsack_pseudo_aggregated(self, mask, n_users, reference_price, noise_alpha, exp_number_noise,
                                             step_size=5,
                                             n_budgets=10):
        """Return knapsack rewards for fully aggregated user classes"""
        noise_alpha = [noise_alpha[0]] + self.noise_alpha
        exp_number_noise = [exp_number_noise[0]] + self.exp_number_noise
        old_budget = self.allocated_budget
        prob_users = [0, 0, 0, 0]
        users = [self.users[0]] + self.users
        for i, bit in enumerate(mask):
            if bit == 1:
                prob_users[i] = self.all_prob_users[i]
        available_budget = [step_size * (i + 1) for i in range(n_budgets)]
        n_campaigns = len(self.products)
        rewards = []
        for _ in range(n_campaigns):
            rewards.append(-1 * np.array(available_budget.copy()))
        rewards = np.array(rewards)

        for cmp_index in range(n_campaigns):
            for budget_idx in range(n_budgets):
                for user_idx, user in enumerate(users):
                    if prob_users[user_idx] != 0:
                        self.campaigns[cmp_index].change_budget(float(available_budget[budget_idx]) * prob_users[
                            user_idx] / float(sum(prob_users)))  # scale allocated budget by probability of user
                        alpha = self.campaigns[cmp_index].get_alpha_i(user.alpha_functions[cmp_index]) * \
                                noise_alpha[user_idx][cmp_index]
                        value_per_click = user.expected_profit(exp_number_noise[user_idx])[cmp_index]
                        expected_gross_profit = prob_users[
                                                    user_idx] * alpha * value_per_click * n_users * reference_price
                        rewards[cmp_index][budget_idx] += np.single(expected_gross_profit)

        self.__set_campaign_budgets(old_budget)  # restore old budget
        return rewards, available_budget

    def __profit_campaign_per_user(self, user_index, n_users, reference_price):
        """ Not scaled user probability profit for a campaign with forced budget of caller"""
        noise_alpha = self.noise_alpha
        exp_number_noise = self.exp_number_noise
        campaign_profits = []
        u = self.users[user_index]

        for i, cmp in enumerate(self.campaigns):

            noise_a = noise_alpha[user_index][i]  # noise over (i,j)
            alpha_f_res = cmp.get_alpha_i(u.alpha_functions[i])  # effect of budget over campaign (alpha function)
            exp_profit = u.expected_profit(exp_number_noise[user_index])[
                i]  # expected profit of user j over graph + noise n effect
            profit_u = noise_a * alpha_f_res * exp_profit  # expected profit of campaign not scaled by user probability

            campaign_profits.append(profit_u)

        # convert the pure number in euro
        for i, cmp_profit in enumerate(campaign_profits):
            euro_val = cmp_profit * n_users * reference_price
            campaign_profits[i] = euro_val

        return campaign_profits

    def __set_user_graph(self, index, graph):
        self.users[index].change_graph(graph)
        self.graphs[index] = graph

    def set_user_graphs(self, g):
        for i, graph in enumerate(g):
            self.__set_user_graph(i, graph)

    def run_graph_estimate(self,
                           numOfEpisodes=100,
                           simulations=100,
                           delta=0.2,  # higher delta, fewer simulations
                           epsilon=0.1,
                           seeds=1,
                           silent=True):
        true_result_history = []
        estimation_fully_con = []
        estimation_2_neigh = []

        for user in self.users:
            """ Graph weights estimation simualation - user 1 - OFFLINE LEARNING - fully connected graph"""
            adjacency_matrix = np.array(user.weighted_graph.get_adjacency_matrix())
            n_campaigns = len(self.campaigns)

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

            ecommerceGraph = util.get_ecommerce_graph(products=self.products)
            estimatedGraph = OnlineWeightsLearner.estimate_weights(true_graph=ecommerceGraph,
                                                                   simulations=simulations,
                                                                   monte_carlo_repetitions=monte_carlo_repetitions)
            estimation_2_neigh.append(estimatedGraph)
            true_result_history.append(ecommerceGraph)
            if not silent:
                print("\nTrue Probability Matrix: \n",
                      util.get_prettyprint_array(ecommerceGraph.get_adjacency_matrix()))
                print("\nEstimated Probability Matrix: \n",
                      util.get_prettyprint_array(estimatedGraph.get_adjacency_matrix()))
                print("\nBetas Matrix: \n", util.get_prettyprint_array(estimatedGraph.get_betas_matrix()))
        return estimation_fully_con, estimation_2_neigh, true_result_history
