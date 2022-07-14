import copy
from typing import Type
from typing import Union
from Environment import Environment
from typing import List
import numpy as np
import pandas as pd
from knapsack.Knapsack import Knapsack
from entities import Utils as util
import matplotlib.pyplot as plt
from learners.CombWrapper import CombWrapper
from tqdm import tqdm
import matplotlib
matplotlib.use("TkAgg")


class SimulationHandler:

    def __init__(self,
                 environmentConstructor: Type[Environment],
                 learners: List['CombWrapper'],
                 experiments: int,
                 days: int,
                 reference_price: Union[float, int],
                 daily_budget: Union[float, int],
                 n_arms: int,
                 n_users: int,
                 bool_alpha_noise: bool,
                 bool_n_noise: bool,
                 print_basic_debug: bool,
                 print_knapsack_info: bool,
                 step_k: int,
                 non_stationary_args: dict = None,
                 is_unknown_graph: bool = False,
                 clairvoyant_type: str = 'aggregated',
                 boost_start: bool = False,
                 boost_discount: float = 0.5,
                 boost_bias: float = -1.0,
                 plot_regressor_progress = None):
        self.environmentConstructor = environmentConstructor
        self.environment = self.environmentConstructor()
        self.learners = learners
        self.learners_rewards_per_experiment = [[] for _ in range(len(self.learners))]
        self.learners_rewards_per_day = [[] for _ in range(len(self.learners))]
        self.super_arms = []
        self.clairvoyant_rewards_per_experiment_t1 = []
        self.clairvoyant_rewards_per_experiment_t2 = []
        self.clairvoyant_rewards_per_day_t1 = []
        self.clairvoyant_rewards_per_day_t2 = []
        self.days = days
        self.experiments = experiments
        self.reference_price = reference_price
        self.daily_budget = daily_budget
        self.n_arms = n_arms
        self.bool_alpha_noise = bool_alpha_noise
        self.bool_n_noise = bool_n_noise
        self.print_basic_debug = print_basic_debug
        self.print_knapsack_info = print_knapsack_info
        self.non_stationary_env = False
        self.n_users = n_users
        self.is_unknown_graph = is_unknown_graph
        self.clairvoyant_type = clairvoyant_type
        self.step_k = step_k
        self.plot_regressor_progress = plot_regressor_progress

        self.boost_start = boost_start
        self.boost_discount = boost_discount
        campaigns = 15 if self.clairvoyant_type == 'disaggregated' else 5  # to generalize !!
        self.boost_bias = boost_bias if boost_bias >= 0.0 else self.daily_budget / campaigns

        if non_stationary_args and isinstance(non_stationary_args, dict):
            self.phase_size = non_stationary_args['phase_size']
            self.num_users_phases = non_stationary_args['num_users_phases']
            self.prob_users_phases = non_stationary_args['prob_users_phases']
            self.non_stationary_env = True

        if self.is_unknown_graph:
                #   **** MONTE CARLO EXECUTION BEFORE SIMULATION ****
                users, products, campaigns, allocated_budget, prob_users, real_graphs = self.environment.get_core_entities()
                self.real_graphs = copy.deepcopy(real_graphs)
                estimated_fully_conn_graphs, estimated_2_neighs_graphs, true_2_neighs_graphs = self.environment.run_graph_estimate()
                self.estimated_fully_conn_graphs = estimated_fully_conn_graphs
                #   ************************************************

    def __set_budgets_env(self, budgets):
        for i, b in enumerate(budgets):
            self.environment.set_campaign_budget(i, b)

    def run_simulation(self):

        self.learners_rewards_per_experiment = [[] for _ in range(len(self.learners))]
        self.clairvoyant_rewards_per_experiment_t1 = []

        if self.clairvoyant_type == 'both':
            self.clairvoyant_rewards_per_experiment_t2 = []

        if self.plot_regressor_progress:
            img, axss = plt.subplots(nrows = 2, ncols = 3, figsize = (13, 6))
            axs = axss.flatten()
            plt.subplots_adjust(hspace = 0.8, top = 0.8)

            for idx, learner in enumerate(self.learners):
                if learner.bandit_name == self.plot_regressor_progress:
                    learner_to_observe = learner
                    idx_learner_to_observe = idx
                    break

            colors = util.get_colors()

        for experiment in range(self.experiments):

            if experiment > 0:
                util.clear_output()

            if True:
                print(f"\n***** EXPERIMENT {experiment + 1} *****")

            self.super_arms = []

            for index, learner in enumerate(self.learners):
                learner.reset()
                self.super_arms.append(learner.pull_super_arm())

            self.learners_rewards_per_day = [[] for _ in range(len(self.learners))]

            self.clairvoyant_rewards_per_day_t1 = []

            if self.clairvoyant_type == 'both':
                self.clairvoyant_rewards_per_day_t2 = []

            self.learners_rewards_per_day = [[] for _ in range(len(self.learners))]

            for day in tqdm(range(self.days)):

                if self.is_unknown_graph:
                    self.environment.set_user_graphs(self.real_graphs)  # set real real_graphs for clavoyrant algorithm

                if self.non_stationary_env:
                    current_phase = int(day / self.phase_size)
                    self.n_users = self.num_users_phases[current_phase]
                    self.environment.prob_users = self.prob_users_phases[current_phase]

                if self.print_basic_debug:
                    print(f"\n***** DAY {day + 1} *****")

                users, products, campaigns, allocated_budget, prob_users, _ = self.environment.get_core_entities()

                sim_obj = self.environment.play_one_day(self.n_users, self.reference_price, self.daily_budget,
                                                        self.step_k,
                                                        self.bool_alpha_noise,
                                                        self.bool_n_noise)  # object with all the day info

                if self.clairvoyant_type == 'both':

                    # AGGREGATED
                    rewards, available_budget = sim_obj["reward_k_agg"]
                    row_label_rewards, row_labels_dp_table, col_labels = util.table_metadata(len(products), 1,
                                                                                             available_budget)

                    K = Knapsack(rewards = rewards, budgets = np.array(available_budget))
                    K.init_for_pretty_print(row_labels = row_labels_dp_table, col_labels = col_labels)
                    K.solve()
                    arg_max = np.argmax(K.get_output()[0][-1])
                    alloc = K.get_output()[1][-1][arg_max]
                    reward = K.get_output()[0][-1][arg_max]
                    self.clairvoyant_rewards_per_day_t1.append(reward)

                    # DISAGGREGATED

                    rewards, available_budget = sim_obj["reward_k"]
                    row_label_rewards, row_labels_dp_table, col_labels = util.table_metadata(len(products), 1,
                                                                                             available_budget)

                    K = Knapsack(rewards = rewards, budgets = np.array(available_budget))
                    K.init_for_pretty_print(row_labels = row_labels_dp_table, col_labels = col_labels)
                    K.solve()
                    arg_max = np.argmax(K.get_output()[0][-1])
                    alloc = K.get_output()[1][-1][arg_max]
                    reward = K.get_output()[0][-1][arg_max]
                    self.clairvoyant_rewards_per_day_t2.append(reward)

                else:
                    rewards, available_budget = sim_obj["reward_k_agg"] if self.clairvoyant_type == 'aggregated' else \
                        sim_obj["reward_k"]

                    row_label_rewards, row_labels_dp_table, col_labels = util.table_metadata(len(products), 1,
                                                                                             available_budget)

                    K = Knapsack(rewards = rewards, budgets = np.array(available_budget))
                    K.init_for_pretty_print(row_labels = row_labels_dp_table, col_labels = col_labels)
                    K.solve()
                    arg_max = np.argmax(K.get_output()[0][-1])
                    alloc = K.get_output()[1][-1][arg_max]
                    reward = K.get_output()[0][-1][arg_max]
                    self.clairvoyant_rewards_per_day_t1.append(reward)

                if self.print_basic_debug:
                    print("\n Aggregated")
                    print(f"best allocation: {alloc}, total budget: {sum(alloc)}")
                    print(f"reward: {reward}")
                    self.__set_budgets_env(alloc[1:])
                    sim_obj_2 = self.environment.replicate_last_day(self.n_users,
                                                                    self.reference_price,
                                                                    self.bool_n_noise,
                                                                    self.bool_n_noise)  # object with all the day info
                    profit1, profit2, profit3, daily_profit = sim_obj_2["profit"]
                    print(
                            f"test allocation on env:\n\t || total:{daily_profit:.2f}€ || u1:{profit1:.2f}€ || u2:{profit2:.2f}€ || u3:{profit3:.2f}€")
                    print("-" * 10 + " Independent rewards Table " + "-" * 10)
                    print(pd.DataFrame(rewards, columns = col_labels, index = row_label_rewards))
                    print("\n" + "*" * 25 + " Aggregated knapsack execution " + "*" * 30 + "\n")
                    K.pretty_print_dp_table()  # prints the final dynamic programming table
                    """K.pretty_print_output(
                        print_last_row_only=False)"""  # prints information about last row of the table, including allocations

                # -----------------------------------------------------------------

                if self.is_unknown_graph:
                    print("ciao")
                    self.environment.set_user_graphs(
                            self.estimated_fully_conn_graphs)  # set real real_graphs for clavoyrant algorithm

                for learnerIdx, learner in enumerate(self.learners):
                    # update with data from today for tomorrow
                    super_arm = self.super_arms[learnerIdx]
                    self.__set_budgets_env(super_arm)
                    # test result on env
                    sim_obj = self.environment.replicate_last_day(self.n_users,
                                                                  self.reference_price,
                                                                  self.bool_n_noise,
                                                                  self.bool_n_noise)

                    profit_env = sim_obj["profit_campaign"][-1]

                    self.learners_rewards_per_day[learnerIdx].append(profit_env - np.sum(super_arm))
                    profit_list = list(sim_obj["profit_campaign"][: -1])

                    # BOOST DONE ONLY TO LEARNERS USING GP REGRESSOR
                    if self.boost_start and learner.needs_boost and day <= 4:
                        for i, s_arm in enumerate(super_arm):
                            if s_arm == 0:
                                # if a learner is pulling 0 give an high reward in an higher arm
                                back_offset = np.random.randint(1, 4)
                                forced_arm = np.sort(super_arm, axis = None)[-back_offset]  # take random high arm value
                                profit_list[i] = np.max(profit_list) * self.boost_discount + self.boost_bias
                                super_arm[i] = forced_arm

                    learner.update_observations(super_arm, profit_list)
                    # solve comb problem for tomorrow
                    self.super_arms[learnerIdx] = learner.pull_super_arm()

                if self.plot_regressor_progress:
                    axs[5].cla()
                if self.plot_regressor_progress:
                    x = available_budget
                    x2 = learner_to_observe.arms
                    for i, rw in enumerate(rewards):
                        axs[i].cla()
                        axs[i].set_xlabel("budget")
                        axs[i].set_ylabel("profit")
                        axs[i].plot(x, rw, colors[-1], label = 'clairvoyant profit')
                        # axs[i].plot(x2, comb_learner.last_knapsack_reward[i])
                        mean, std = learner_to_observe.get_gp_data()
                        # print(std[0])
                        # print(mean[0][0])
                        axs[i].plot(x2, mean[i], colors[i], label = 'estimated profit')
                        axs[i].fill_between(
                                np.array(x2).ravel(),
                                mean[i] - 1.96 * std[i],
                                mean[i] + 1.96 * std[i],
                                alpha = 0.5,
                                label = r"95% confidence interval",
                        )

                        axs[i].legend(bbox_to_anchor = (0., 1.02, 1., .102), loc = 3,
                                      ncol = 2, mode = "expand", borderaxespad = 0.)

                    d = np.linspace(0, len(self.clairvoyant_rewards_per_day_t1),
                                    len(self.clairvoyant_rewards_per_day_t1))
                    axs[5].set_xlabel("days")
                    axs[5].set_ylabel("reward")
                    axs[5].plot(d, self.clairvoyant_rewards_per_day_t1, colors[-1], label = "clairvoyant reward")
                    axs[5].plot(d, self.learners_rewards_per_day[idx_learner_to_observe], colors[-2],label = "bandit reward")
                    axs[5].legend(bbox_to_anchor = (0., 1.02, 1., .102), loc = 3,
                                  ncol = 2, mode = "expand", borderaxespad = 0.)

                    # axs[5].plot(d, rewards_disaggregated)
                    plt.pause(0.02)  # no need for this,

            self.clairvoyant_rewards_per_experiment_t1.append(self.clairvoyant_rewards_per_day_t1)

            if self.clairvoyant_type == 'both':
                self.clairvoyant_rewards_per_experiment_t2.append(self.clairvoyant_rewards_per_day_t2)

            for learnerIdx in range(len(self.learners)):
                self.learners_rewards_per_experiment[learnerIdx].append(self.learners_rewards_per_day[learnerIdx])

        self.__plot_results()

    # TODO A PLOT HANDLER SHOULD DO ALL THE WORK HERE !

    def __plot_results(self):

        clairvoyant_rewards_per_experiment_t1 = np.array(self.clairvoyant_rewards_per_experiment_t1)

        clairvoyant_rewards_per_experiment_t2 = np.array(self.clairvoyant_rewards_per_experiment_t2)
        learners_rewards_per_experiment = np.array(self.learners_rewards_per_experiment)

        if self.clairvoyant_type != 'both':
            print(f"\n***** FINAL RESULT CLAIRVOYANT ALGORITHM {self.clairvoyant_type.upper()} *****")
            print(f"days simulated: {self.days}")
            print(
                    f"average clairvoyant total profit:\t {float(np.mean(np.sum(clairvoyant_rewards_per_experiment_t1, axis = 1))):.4f}€")
            print(f"average clairvoyant profit per day:\t {float(np.mean(clairvoyant_rewards_per_experiment_t1)):.4f}€")
            print(f"average standard deviation:\t {float(np.std(clairvoyant_rewards_per_experiment_t1)):.4f}€")

        else:

            print(f"\n***** FINAL RESULT CLAIRVOYANT ALGORITHM DISAGGREGATED *****")
            print(f"days simulated: {self.days}")
            print(
                    f"average clairvoyant total profit:\t {float(np.mean(np.sum(clairvoyant_rewards_per_experiment_t2, axis = 1))):.4f}€")
            print(f"average clairvoyant profit per day:\t {float(np.mean(clairvoyant_rewards_per_experiment_t2)):.4f}€")
            print(f"average standard deviation:\t {float(np.std(clairvoyant_rewards_per_experiment_t2)):.4f}€")

            print(f"\n***** FINAL RESULT CLAIRVOYANT ALGORITHM AGGREGATED *****")
            print(f"days simulated: {self.days}")
            print(
                    f"average clairvoyant total profit:\t {float(np.mean(np.sum(clairvoyant_rewards_per_experiment_t1, axis = 1))):.4f}€")
            print(f"average clairvoyant profit per day:\t {float(np.mean(clairvoyant_rewards_per_experiment_t1)):.4f}€")
            print(f"average standard deviation:\t {float(np.std(clairvoyant_rewards_per_experiment_t1)):.4f}€")

        for learnerIdx, learner in enumerate(self.learners):
            print(f"\n***** FINAL RESULT LEARNER {learner.bandit_name} *****")
            print(
                    f"Learner average total profit:\t {float(np.mean(np.sum(learners_rewards_per_experiment[learnerIdx], axis = 1))):.4f}€")
            print("----------------------------")
            print(
                    f"Learner average profit per day :\t {float(np.mean(learners_rewards_per_experiment[learnerIdx])):.4f}€")
            print(f"average standard deviation:\t {float(np.std(learners_rewards_per_experiment[learnerIdx])):.4f}€")
            print(
                    f"average regret\t {float(np.mean(clairvoyant_rewards_per_experiment_t1 - learners_rewards_per_experiment[learnerIdx])):.4f}€")
            print(
                    f"average standard deviation:\t {float(np.std(clairvoyant_rewards_per_experiment_t1 - learners_rewards_per_experiment[learnerIdx])):.4f}€")
            print()

        plt.close('all')

        d = np.linspace(0, self.days, self.days)


        colors = util.get_colors()

        colors_learners = [colors.pop() for _ in range(len(self.learners))]

        if len(self.learners) > 0:
            img, axss = plt.subplots(nrows = 2, ncols = 2, figsize = (13, 6))
            plt.subplots_adjust(hspace = 0.8, top = 0.8)
        else:
            img, axss = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 6))

        axs = axss.flatten()

        axs[0].set_xlabel("days")
        axs[0].set_ylabel("reward")

        if self.clairvoyant_type != 'both':
            axs[0].plot(d, np.mean(clairvoyant_rewards_per_experiment_t1, axis = 0), colors[-1],
                        label = "clairvoyant")

        else:
            axs[0].plot(d, np.mean(clairvoyant_rewards_per_experiment_t1, axis = 0), colors[-1],
                        label = "clairvoyant aggregated")
            axs[0].plot(d, np.mean(clairvoyant_rewards_per_experiment_t2, axis = 0), colors[-2],
                        label = "clairvoyant disaggregated")

        axs[1].set_xlabel("days")
        axs[1].set_ylabel("cumulative reward")

        for learnerIdx in range(len(self.learners)):
            bandit_name = self.learners[learnerIdx].bandit_name
            axs[0].plot(d, np.mean(learners_rewards_per_experiment[learnerIdx], axis = 0), colors_learners[learnerIdx],
                        label = bandit_name)

        if self.clairvoyant_type != 'both':
            axs[1].plot(d, np.cumsum(np.mean(clairvoyant_rewards_per_experiment_t1, axis = 0)), colors[-1],
                        label = "clairvoyant")

        else:
            axs[1].plot(d, np.cumsum(np.mean(clairvoyant_rewards_per_experiment_t1, axis = 0)), colors[-1],
                        label = "clairvoyant aggregated")
            axs[1].plot(d, np.cumsum(np.mean(clairvoyant_rewards_per_experiment_t2, axis = 0)), colors[-2],
                        label = "clairvoyant disaggegated")

        for learnerIdx in range(len(self.learners)):
            bandit_name = self.learners[learnerIdx].bandit_name
            axs[1].plot(d, np.cumsum(np.mean(learners_rewards_per_experiment[learnerIdx], axis = 0)),
                        colors_learners[learnerIdx], label = bandit_name)

        axs[0].legend(bbox_to_anchor = (0., 1.02, 1., .102), loc = 3,
                      ncol = 2, mode = "expand", borderaxespad = 0.)
        axs[1].legend(bbox_to_anchor = (0., 1.02, 1., .102), loc = 3,
                      ncol = 2, mode = "expand", borderaxespad = 0.)

        if len(self.learners) > 0:
            axs[2].set_xlabel("days")
            axs[2].set_ylabel("cumulative regret")

            axs[3].set_xlabel("days")
            axs[3].set_ylabel("regret")

            for learnerIdx in range(len(self.learners)):
                bandit_name = self.learners[learnerIdx].bandit_name
                axs[2].plot(d, np.cumsum(
                        np.mean(clairvoyant_rewards_per_experiment_t1 - learners_rewards_per_experiment[learnerIdx],
                                axis = 0)),
                            colors_learners[learnerIdx], label = bandit_name)

            for learnerIdx in range(len(self.learners)):
                bandit_name = self.learners[learnerIdx].bandit_name
                axs[3].plot(d,
                            np.mean(clairvoyant_rewards_per_experiment_t1 - learners_rewards_per_experiment[learnerIdx],
                                    axis = 0), colors_learners[learnerIdx], label = bandit_name)

            axs[2].legend(bbox_to_anchor = (0., 1.02, 1., .102), loc = 3,
                          ncol = 2, mode = "expand", borderaxespad = 0.)
            axs[3].legend(bbox_to_anchor = (0., 1.02, 1., .102), loc = 3,
                          ncol = 2, mode = "expand", borderaxespad = 0.)

        plt.show()
