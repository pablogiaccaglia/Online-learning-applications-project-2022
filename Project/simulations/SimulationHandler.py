import copy
import seaborn as sns
from typing import Type
from typing import Union
from typing import List
import numpy as np
from entities import Utils as util
import matplotlib.pyplot as plt
from learners.CombWrapper import CombWrapper
from tqdm import tqdm
import matplotlib

from simulations.Environment import Environment

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
            plt.subplots_adjust(left = 0.05, right = 0.95, hspace = 0.6, top = 0.9, wspace = 0.4, bottom = 0.1)

            for ax in axs:
                ax.grid(alpha = 0.2)

            sns.set_style("ticks")
            sns.despine()
            sns.set_context('notebook')

            learner_to_observe = None

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
                    self.clairvoyant_rewards_per_day_t1.append(sim_obj["reward_k_agg"])

                    # DISAGGREGATED
                    self.clairvoyant_rewards_per_day_t2.append(sim_obj["reward_k_disagg"])

                else:
                    reward = sim_obj["reward_k_agg"] if self.clairvoyant_type == 'aggregated' else sim_obj[
                        "reward_k_disagg"]
                    self.clairvoyant_rewards_per_day_t1.append(reward)

                # -----------------------------------------------------------------

                if self.is_unknown_graph:
                    """print("Estimated Graph setted")"""
                    self.environment.set_user_graphs(
                            self.estimated_fully_conn_graphs)  # set real real_graphs for clavoyrant algorithm

                for learnerIdx, learner in enumerate(self.learners):
                    # update with data from today for tomorrow
                    super_arm = self.super_arms[learnerIdx]

                    # BOOST (Random exploration) DONE ONLY TO LEARNERS USING GP REGRESSOR
                    if self.boost_start and learner.needs_boost and day < 4:
                        idx = np.random.choice(len(learner.arms) - 1, 5, replace=True)
                        loop = 0
                        while np.sum(np.array(learner.arms)[idx]) >= self.daily_budget:
                            idx = np.random.choice(len(learner.arms) - 1 - loop, 5, replace=True)
                            loop += 1
                        # force random exploration
                        super_arm = np.array(learner.arms)[idx]

                    sim_obj_2 = self.environment.replicate_last_day(super_arm,
                                                                    self.n_users,
                                                                    self.reference_price,
                                                                    self.bool_n_noise,
                                                                    self.bool_n_noise)
                    profit_env = sim_obj_2["profit"]
                    learner_rewards = sim_obj_2["learner_rewards"]
                    net_profit_learner = np.sum(learner_rewards)
                    learner.update_observations(super_arm, learner_rewards)

                    # solve comb problem for tomorrow
                    self.super_arms[learnerIdx] = learner.pull_super_arm()

                    self.learners_rewards_per_day[learnerIdx].append(net_profit_learner)

                if self.plot_regressor_progress and learner_to_observe:
                    axs[5].cla()
                    x = sim_obj["k_budgets"]
                    x2 = learner_to_observe.arms
                    for i, rw in enumerate(sim_obj["rewards_agg"]):
                        axs[i].cla()
                        axs[i].set_xlabel("budget")
                        axs[i].set_ylabel("profit")
                        axs[i].plot(x, rw, colors[-1], label = 'clairvoyant profit', alpha = 0.5)
                        # axs[i].plot(x2, comb_learner.last_knapsack_reward[i])
                        mean, std = learner_to_observe.get_gp_data()
                        # print(std[0])
                        # print(mean[0][0])
                        axs[i].plot(x2, mean[i], colors[i], label = 'estimated profit', alpha = 0.5)
                        axs[i].fill_between(
                                np.array(x2).ravel(),
                                mean[i] - 1.96 * std[i],
                                mean[i] + 1.96 * std[i],
                                alpha = 0.1,
                                label = r"95% confidence interval",
                                color = colors[i]
                        )

                        axs[i].legend(bbox_to_anchor = (0., 1.02, 1., .102), loc = 3,
                                      ncol = 2, mode = "expand", borderaxespad = 0.)

                        axs[i].set_title('Profit curve - Campaign ' + str(i + 1), y = 1.0, pad = 43)

                    d = np.linspace(0, len(self.clairvoyant_rewards_per_day_t1),
                                    len(self.clairvoyant_rewards_per_day_t1))
                    axs[5].set_xlabel("days")
                    axs[5].set_ylabel("reward")
                    axs[5].plot(d, self.clairvoyant_rewards_per_day_t1, colors[-1], label = "clairvoyant reward",
                                alpha = 0.5)
                    axs[5].plot(d, self.learners_rewards_per_day[idx_learner_to_observe], colors[-2],
                                label = "bandit reward", alpha = 0.5)
                    axs[5].legend(bbox_to_anchor = (0., 1.02, 1., .102), loc = 3,
                                  ncol = 2, mode = "expand", borderaxespad = 0.)

                    axs[5].set_title('Reward', y = 1.0, pad = 28)

                    # axs[5].plot(d, rewards_disaggregated)
                    plt.pause(0.02)  # no need for this,

            self.clairvoyant_rewards_per_experiment_t1.append(self.clairvoyant_rewards_per_day_t1)

            if self.clairvoyant_type == 'both':
                self.clairvoyant_rewards_per_experiment_t2.append(self.clairvoyant_rewards_per_day_t2)

            for learnerIdx in range(len(self.learners)):
                self.learners_rewards_per_experiment[learnerIdx].append(self.learners_rewards_per_day[learnerIdx])

        # self.__plot_results(sns_style = 'white') # looks nice

        # self.__plot_results(sns_style = 'black') # looks nice but i do not it like that much
        self.__plot_results(sns_style = 'matplotlib')  # uses default matplotlib style

    # TODO A PLOT HANDLER SHOULD DO ALL THE WORK HERE !
    # TODO -> PLOT CONFIDENCE INTERVALS !

    def __plot_results(self, remove_splines = True, offset_axes = True, opacity = 0.5, sns_context = 'notebook',
                       set_ticks = False, sns_style = 'matplotlib', enable_grid = True, hspace = 1, wspace = 0.5):

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

        sns.set_context(context = sns_context)
        if sns_style != 'matplotlib':
            sns.set_style(style = sns_style)

        if set_ticks:
            sns.set_style("ticks")

        d = np.linspace(0, self.days, self.days)

        colors = util.get_colors()

        colors_learners = [colors.pop() for _ in range(len(self.learners))]

        if len(self.learners) > 0:
            img, axss = plt.subplots(nrows = 2, ncols = 2, figsize = (13, 6))
            plt.subplots_adjust(hspace = hspace, top = 0.8, wspace = wspace)
        else:
            img, axss = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 6))

        axs = axss.flatten()

        axs[0].set_xlabel("days")
        axs[0].set_ylabel("reward")

        if self.clairvoyant_type != 'both':
            axs[0].plot(d, np.mean(clairvoyant_rewards_per_experiment_t1, axis = 0), colors[-1],
                        label = "clairvoyant", alpha = opacity)

        else:
            axs[0].plot(d, np.mean(clairvoyant_rewards_per_experiment_t1, axis = 0), colors[-1],
                        label = "clairvoyant aggregated", alpha = opacity)
            axs[0].plot(d, np.mean(clairvoyant_rewards_per_experiment_t2, axis = 0), colors[-2],
                        label = "clairvoyant disaggregated", alpha = opacity)

        axs[1].set_xlabel("days")
        axs[1].set_ylabel("cumulative reward")

        for learnerIdx in range(len(self.learners)):
            bandit_name = self.learners[learnerIdx].bandit_name
            axs[0].plot(d, np.mean(learners_rewards_per_experiment[learnerIdx], axis = 0), colors_learners[learnerIdx],
                        label = bandit_name, alpha = opacity)

        if self.clairvoyant_type != 'both':
            axs[1].plot(d, np.cumsum(np.mean(clairvoyant_rewards_per_experiment_t1, axis = 0)), colors[-1],
                        label = "clairvoyant", alpha = opacity)

        else:
            axs[1].plot(d, np.cumsum(np.mean(clairvoyant_rewards_per_experiment_t1, axis = 0)), colors[-1],
                        label = "clairvoyant aggregated", alpha = opacity)
            axs[1].plot(d, np.cumsum(np.mean(clairvoyant_rewards_per_experiment_t2, axis = 0)), colors[-2],
                        label = "clairvoyant disaggegated", alpha = opacity)

        for learnerIdx in range(len(self.learners)):
            bandit_name = self.learners[learnerIdx].bandit_name
            axs[1].plot(d, np.cumsum(np.mean(learners_rewards_per_experiment[learnerIdx], axis = 0)),
                        colors_learners[learnerIdx], label = bandit_name, alpha = opacity)

        axs[0].legend(bbox_to_anchor = (0., 1.02, 1., .102), loc = 3,
                      ncol = 2, mode = "expand", borderaxespad = 0.)
        axs[1].legend(bbox_to_anchor = (0., 1.02, 1., .102), loc = 3,
                      ncol = 2, mode = "expand", borderaxespad = 0.)

        if len(self.learners) > 0:
            axs[3].set_xlabel("days")
            axs[3].set_ylabel("cumulative regret")

            axs[2].set_xlabel("days")
            axs[2].set_ylabel("regret")

            for learnerIdx in range(len(self.learners)):
                bandit_name = self.learners[learnerIdx].bandit_name
                axs[3].plot(d, np.cumsum(
                        np.mean(clairvoyant_rewards_per_experiment_t1 - learners_rewards_per_experiment[learnerIdx],
                                axis = 0)),
                            colors_learners[learnerIdx], label = bandit_name, alpha = opacity)

            for learnerIdx in range(len(self.learners)):
                bandit_name = self.learners[learnerIdx].bandit_name
                axs[2].plot(d,
                            np.mean(clairvoyant_rewards_per_experiment_t1 - learners_rewards_per_experiment[learnerIdx],
                                    axis = 0), colors_learners[learnerIdx], label = bandit_name, alpha = opacity)

            axs[3].legend(bbox_to_anchor = (0., 1.02, 1., .102), loc = 3,
                          ncol = 2, mode = "expand", borderaxespad = 0.)
            axs[2].legend(bbox_to_anchor = (0., 1.02, 1., .102), loc = 3,
                          ncol = 2, mode = "expand", borderaxespad = 0.)

        if remove_splines:
            if offset_axes:
                for ax in axs:
                    sns.despine(ax = ax, offset = 5, trim = False)
            else:
                sns.despine()

        if enable_grid:
            for ax in axs:
                ax.grid(alpha = 0.2)

        # REALLY BAD CODE HERE !!!!!!

        if len(self.learners) > 1:
            pad = 45
        elif len(self.learners) > 0:
            pad = 28
        else:
            pad = 36

        axs[0].set_title('Reward', y = 1.0, pad = pad)
        axs[1].set_title('Cumulative Reward', y = 1.0, pad = pad)

        if len(self.learners) > 0:
            if len(self.learners) > 2:
                pad = 46
            elif len(self.learners) > 0:
                pad = 28

            axs[3].set_title('Cumulative Regret', y = 1.0, pad = pad)
            axs[2].set_title('Regret', y = 1.0, pad = pad)

        plt.show()
