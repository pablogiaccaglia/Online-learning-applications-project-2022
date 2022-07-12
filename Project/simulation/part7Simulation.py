import numpy as np
import pandas as pd
from Knapsack import Knapsack
from Part3.CombWrapper import CombWrapper
from Part3.GPTS_Learner import GPTS_Learner
from Part3.GTS_Learner import GTS_Learner
from simulation.Environment import Environment
import matplotlib.pyplot as plt
import progressbar

""" @@@@ simulation SETUP @@@@ """

"""
+---------+---------+--------+
|    .    | Student | Worker |
+---------+---------+--------+
| Family  | usr1    | usr1   |
| Alone   | usr2    | usr3   |
|         |         |        |
+---------+---------+--------+
"""
days = 160
N_user = 400  # reference for what alpha = 1 refers to
reference_price = 4.0
daily_budget = 70 * 5
n_arms = 20
step_k = 4
context_gen_days = 40
interval = 2  # how often to run context gen.
n_budget = int(daily_budget / step_k)
environment = Environment()

bool_alpha_noise = False
bool_n_noise = False
printBasicDebug = False
printKnapsackInfo = False

# distribute uniformely user1 worker and user1 split_student
p_usr1_s = environment.prob_users[0] * 0.5
p_usr1_w = environment.prob_users[0] * 0.5
p_usr2 = environment.prob_users[1]
p_usr3 = environment.prob_users[2]
p_users = [p_usr1_s, p_usr1_w, p_usr2, p_usr3]
""" @@@@ ---------------- @@@@ """
# ******* Context initialization ********
rewards_clairvoyant = []
base_learner_rewards = []
ctx_learner_rewards = []
budgets_array = np.array([
    [1, 1, 1, 1, 1],  # u 11
    [1, 1, 1, 1, 1],  # u 12
    [1, 1, 1, 1, 1],  # u 2
    [1, 1, 1, 1, 1]  # u3
]) * 50

context_on = False
target_feature = [True, False]  # start fully aggregated
stop_context = False
context_initialized = False

ctx_algorithm = GPTS_Learner
# ctx_algorithm = GTS_Learner


def table_metadata(n_prod, n_users, avail_budget):
    _col_labels = [str(budget) for budget in avail_budget]

    _row_label_rewards = []
    _row_labels_dp_table = ['0']
    for i in range(1, n_prod + 1):
        for j in range(1, n_users + 1):
            # Cij -> campaign i and user j
            _row_label_rewards.append("C" + str(i) + str(j))
            _row_labels_dp_table.append("+C" + str(i) + str(j))
    return _row_label_rewards, _row_labels_dp_table, _col_labels


def set_budgets_knapsack_env(knapsack_alloc):
    # add knapsack offset
    for i, b in enumerate(knapsack_alloc[1:]):
        environment.set_campaign_budget(i, b)


def set_budgets_arm_env(s_arm):
    for i, b in enumerate(s_arm):
        environment.set_campaign_budget(i, b)


def context_masks(split_family=False, split_student=False):
    # return a mask for the contextual basic blocks
    masks = []
    if not split_family and not split_student:
        return [[1, 1, 1, 1]]

    if split_family and split_student:
        masks.append([1, 0, 0, 0])
        masks.append([0, 1, 0, 0])
        masks.append([0, 0, 1, 0])
        masks.append([0, 0, 0, 1])

        return masks
    if split_family:
        masks.append([1, 1, 0, 0])
        masks.append([0, 0, 1, 1])
    if split_student:
        masks.append([1, 0, 1, 0])
        masks.append([0, 1, 0, 1])
    # [usr1_s, usr1_w, usr2, usr3]
    return masks


def assemble_profit(_profit_blocks, _contexts, _p_users, flatten=False):
    """Perform addition and scale by user probability in the context"""
    assembled_profit = []
    block = np.array(_profit_blocks).T  # transpose profit to (camp x usr)
    for mask in _contexts:
        scaled_mask = np.array(mask) * np.array(_p_users)
        context_profit = block @ scaled_mask
        assembled_profit.append(context_profit)
    if flatten:
        return np.array(assembled_profit).flatten()

    return np.array(assembled_profit)


def budget_array_from_superarm(_super_arm, _contexts, _p_users):
    """ map the super arm result with blocks of budget for every possible context participant """
    # TESTED OK
    if len(_super_arm) / len(_contexts) != 5.0:
        raise ValueError(f"Super arm not compatible with context {len(_super_arm)}/{len(_contexts)} != 5 \n "
                         f"{_super_arm} || {_contexts}")
    budgets = np.array(_super_arm).reshape((len(_contexts), 5))
    result = np.zeros((4, 5))
    for i, ctx in enumerate(_contexts):
        mask = np.array(ctx)
        scaled_mask = mask * np.array(_p_users) / np.sum(mask * np.array(_p_users))
        scaled_mask = scaled_mask.reshape((len(ctx), 1))
        b = budgets[i].reshape((1, 5))
        result += scaled_mask @ b

    return result  # matrix of scaled budgets


def budget_array_from_k_alloc(_alloc, flatten=False):
    """ map the super arm result with blocks of budget for every possible context participant """
    # _alloc = [0, 1, 2, 3, 11, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 333]
    if len(_alloc) != 16:
        raise ValueError("Knapsack disaggregated alloc needed")
    alloc_clean = _alloc[1:]  # remove 0
    a = np.array(alloc_clean).reshape((5, 3))  # reshape in cluster of 3 res x 5 camp
    tmp = np.reshape(a, len(alloc_clean), order='F')  # rorder per user
    second_part = tmp.reshape((3, 5))  # reshape 5 camp x 3 user
    first_part = second_part[0].copy()[None, :]  # add extra dimention for user 1 double
    if flatten:
        return np.concatenate((first_part, second_part), axis=0).flatten()
    return np.concatenate((first_part, second_part), axis=0)


def set_budgets_env(knapsack_alloc):
    for i, b in enumerate(knapsack_alloc[1:]):
        environment.set_campaign_budget(i, b)


# initialize base learner target and shape
contexts = context_masks(split_family=target_feature[0], split_student=target_feature[1])
size_ctx = len(contexts)
start_offset = 0
n_campaigns_ctx = 5 * size_ctx
base_learner = CombWrapper(ctx_algorithm, n_campaigns_ctx, n_arms, daily_budget)
super_arm = base_learner.pull_super_arm()
target_feature_i = 0
last_superarm = super_arm
# ***************************************


img, axss = plt.subplots(nrows=2, ncols=3, figsize=(13, 6))
axs = axss.flatten()

for day in progressbar.progressbar(range(days)):
    # print(f"Current context: {contexts}")
    users, products, campaigns, allocated_budget, prob_users = environment.get_core_entities()
    sim_obj = environment.play_one_day(N_user, reference_price, daily_budget, step_k, bool_alpha_noise,
                                       bool_n_noise)  # object with all the day info
    # disaggregated knapsack   --------------------------------------------
    rewards, available_budget = sim_obj["reward_k"]
    row_label_rewards, row_labels_dp_table, col_labels = table_metadata(len(products), len(users), available_budget)
    K = Knapsack(rewards=rewards, budgets=np.array(available_budget))
    K.init_for_pretty_print(row_labels=row_labels_dp_table, col_labels=col_labels)
    K.solve()

    arg_max = np.argmax(K.get_output()[0][-1])
    alloc = K.get_output()[1][-1][arg_max]
    reward = K.get_output()[0][-1][arg_max]  # reward of K is correct and is net_profit

    # print(f"\nclairvoyant alloc: {budget_array_from_k_alloc(alloc, flatten=True)}")
    b_knap = budget_array_from_k_alloc(alloc)  # budgets vector for contextualized env
    profit_blocks = environment.get_context_building_blocks(budgets_array=b_knap,
                                                            prob_users=p_users,
                                                            n_users=N_user,
                                                            reference_price=reference_price)  # test clairvoyant on env
    gross_profit_k_env = np.sum(assemble_profit(profit_blocks, contexts, p_users))
    net_profit_k_env = gross_profit_k_env - np.sum(alloc)
    # print(f"\nrew clair: {net_profit_k_env}")
    # print(f"budget clair: {np.sum(alloc)}")
    rewards_clairvoyant.append(net_profit_k_env)

    # -------------------------------------------------------------------------
    # if day != 0 and (day + start_offset) % interval == 0:
    if day == 40 or day == 100 or day == 160:
        if target_feature_i <= 1:
            context_on = True
            last_superarm = super_arm  # save arm of possible old learner
            target_feature[target_feature_i] = True  # split for next feature
            target_feature_i += 1
            start_day = day
            print(f"context ON - day {day} bool1: {target_feature[0]} bool2: {target_feature[1]}")

    if context_on:
        # generate contexts list of context masks
        contexts = context_masks(split_family=target_feature[1], split_student=target_feature[0])
        size_ctx = len(contexts)

        if not context_initialized:
            # initialize candidate learner
            n_campaigns_ctx = 5 * size_ctx
            ctx_learner = CombWrapper(ctx_algorithm, n_campaigns_ctx, n_arms, daily_budget)
            # todo add here way to pass mean and variance to new learner (ESSENTIAL TO WORK)
            # todo try to add the bootsrap for first iteration of learning
            super_arm = ctx_learner.pull_super_arm()  # pull arm for new learner
            context_initialized = True

        # generate budget matrix from pulled superarm
        budgets_array = budget_array_from_superarm(super_arm, contexts, p_users)
        # get gross profits from env
        profit_blocks = environment.get_context_building_blocks(budgets_array=budgets_array,
                                                                prob_users=p_users,
                                                                n_users=N_user,
                                                                reference_price=reference_price)
        # gross profit used as reward to the learner
        learner_reward = assemble_profit(profit_blocks, contexts, p_users)

        ctx_learner.update_observations(super_arm, learner_reward)
        # collect net profit of learner (net reward)
        ctx_learner_rewards.append(np.sum(learner_reward) - np.sum(super_arm))
        # todo: decide what to do with htis part, it is provisory to follow the ctx learner
        base_learner_rewards.append(np.sum(learner_reward) - np.sum(super_arm))
        # print(f"Bandit alloc: {super_arm}")

        # pull super arm for tomorrow
        super_arm = ctx_learner.pull_super_arm()
        # print(f"\nctx learner rew : {np.sum(learner_reward) - np.sum(super_arm)}")

        if day % 10 == 0:
            axs[0].cla()
        if day % 2 == 0:
            x = available_budget
            d = np.linspace(0, len(rewards_clairvoyant), len(rewards_clairvoyant))
            axs[0].set_xlabel("days")
            axs[0].set_ylabel("reward")
            axs[0].plot(d, rewards_clairvoyant)
            axs[0].plot(d, base_learner_rewards)
            plt.pause(0.1)

        if day == start_day + context_gen_days:
            # stop generator
            print(f"context OFF - day {day}")
            start_offset = start_day + context_gen_days
            context_on = False
            context_initialized = False
            split_condition = True  # todo: develop here tecnique for splitting + context management
            if split_condition:
                base_learner = ctx_learner
            else:
                super_arm = last_superarm
    else:
        budgets_array = budget_array_from_superarm(super_arm, contexts, p_users)
        # get profits from env (need super arm)
        profit_blocks = environment.get_context_building_blocks(budgets_array=budgets_array,
                                                                prob_users=p_users,
                                                                n_users=N_user,
                                                                reference_price=reference_price)
        learner_reward = assemble_profit(profit_blocks, contexts, p_users)

        """if day <= 6:
            learner_reward = learner_reward.flatten() + super_arm * 1.2"""
        net_profit_learner = np.sum(learner_reward) - np.sum(super_arm)
        base_learner.update_observations(super_arm, learner_reward.flatten())
        base_learner_rewards.append(net_profit_learner)
        # print(f"\nlearner rew : {np.sum(learner_reward) - np.sum(super_arm)}")
        # print(f"budget ler: {np.sum(super_arm)}")
        # ctx_learner_rewards.append(np.sum(learner_reward) - np.sum(super_arm))

        # pull super arm for tomorrow
        super_arm = base_learner.pull_super_arm()

        if day % 10 == 0:
            axs[0].cla()
        if day % 2 == 0:
            x = available_budget
            d = np.linspace(0, len(rewards_clairvoyant), len(rewards_clairvoyant))
            axs[0].set_xlabel("days")
            axs[0].set_ylabel("reward")
            axs[0].plot(d, rewards_clairvoyant)
            axs[0].plot(d, base_learner_rewards)
            plt.pause(0.1)
plt.show()

# ********* statistical measures *********************
print(f"super arm:  {super_arm}")
print(f"alloc knap: {alloc[1:]}")

print(f"\n***** FINAL RESULT *****")
print(f"days simulated: {days}")
print(f"total profit:\t {sum(rewards_clairvoyant):.4f}€")
print(f"standard deviation:\t {np.std(rewards_clairvoyant):.4f}€")
print(base_learner_rewards)
print(f"Learner profit:\t {sum(base_learner_rewards):.4f}€")
print("----------------------------")
print(f"average profit:\t {np.mean(rewards_clairvoyant):.4f}€")
print(f"\tstd:\t {np.std(rewards_clairvoyant):.4f}€")
print(f"average reward:\t {np.mean(base_learner_rewards):.4f}€")
print(f"\tstd:\t {np.std(base_learner_rewards):.4f}€")
print(f"average regret\t {np.mean(np.array(rewards_clairvoyant) - np.array(base_learner_rewards)):.4f}€")
print(f"\tstd:\t {np.std(np.array(rewards_clairvoyant) - np.array(base_learner_rewards)):.4f}€")

plt.close()
d = np.linspace(0, len(rewards_clairvoyant), len(rewards_clairvoyant))

img, axss = plt.subplots(nrows=2, ncols=2, figsize=(13, 6))
axs = axss.flatten()

axs[0].set_xlabel("days")
axs[0].set_ylabel("reward")
axs[0].plot(d, rewards_clairvoyant)
axs[0].plot(d, base_learner_rewards)

axs[1].set_xlabel("days")
axs[1].set_ylabel("cumulative reward")
axs[1].plot(d, np.cumsum(rewards_clairvoyant))
axs[1].plot(d, np.cumsum(base_learner_rewards))

axs[2].set_xlabel("days")
axs[2].set_ylabel("cumulative regret")
axs[2].plot(d, np.cumsum(np.array(rewards_clairvoyant) - np.array(base_learner_rewards)))

axs[3].set_xlabel("days")
axs[3].set_ylabel("regret")
axs[3].plot(d, np.array(rewards_clairvoyant) - np.array(base_learner_rewards))
plt.show()

"""
    print(environment.get_context_building_blocks(budgets_array=budgets_array,
                                                  n_users=N_user,
                                                  reference_price=reference_price,
                                                  step_size=step_k,
                                                  n_budget=n_budget)[0][0])
    print("@_<>_@")
    print(environment.get_context_building_blocks(budgets_array=budgets_array,
                                                  n_users=N_user,
                                                  reference_price=reference_price,
                                                  step_size=step_k,
                                                  n_budget=n_budget)[1])"""

"""def ctx_vectors(_env_context, _target_feature):
    if _target_feature not in [0, 1]:
        raise ValueError("Invalid feature")
    if _env_context[_target_feature] != -1:
        raise ValueError("Feature already considered in env")
    v1 = _env_context.copy()
    v2 = _env_context.copy()
    v1[_target_feature] = 0
    v2[_target_feature] = 1
    return v1, v2


for day in progressbar.progressbar(range(days)):
    users, products, campaigns, allocated_budget, prob_users = environment.get_core_entities()
    sim_obj = environment.play_one_day(N_user, reference_price, daily_budget, bool_alpha_noise,
                                       bool_n_noise)  # object with all the day info

    # disaggregated knapsack   --------------------------------------------
    rewards, available_budget = sim_obj["reward_k"]
    row_label_rewards, row_labels_dp_table, col_labels = table_metadata(len(products), len(users), available_budget)
    K = Knapsack(rewards=rewards, budgets=np.array(available_budget))
    K.init_for_pretty_print(row_labels=row_labels_dp_table, col_labels=col_labels)
    K.solve()

    alloc = K.get_output()[1][-1][-1]
    reward = K.get_output()[0][-1][-1]
    rewards_disaggregated.append(reward)
    # -----------------------------------------------------------------

    # GTS and GPTS   --------------------------------------------
    # update with data from today for tomorrow
    set_budgets_arm_env(super_arm)
    # test result on env
    sim_obj_2 = environment.replicate_last_day(N_user,
                                               reference_price,
                                               bool_n_noise,
                                               bool_n_noise)

    comb_learner.update_observations(super_arm, sim_obj_2["profit_campaign"][:-1])
    learner_rewards.append(sim_obj_2["profit_campaign"][-1] - np.sum(super_arm))
    if not context_on:
        reward_ctx1.append(sim_obj_2["profit_campaign"][-1] - np.sum(super_arm))
        reward_ctx2.append(sim_obj_2["profit_campaign"][-1] - np.sum(super_arm))

    # solve comb problem for tomorrow
    super_arm = comb_learner.pull_super_arm()

    if context_on:
        context_vectors = ctx_vectors(env_context, target_feature)
        print(f"Working on split: {context_vectors}")
        usr_context1 = filtered_users(users, 1, 1)
        usr_context2 = filtered_users(users, 1, 0)"""

"""    # run knapsack
    rewards, available_budget = sim_obj["reward_k_agg"]
    row_label_rewards, row_labels_dp_table, col_labels = table_metadata(len(products), 1, available_budget)
    K = Knapsack(rewards=rewards, budgets=np.array(available_budget))
    K.init_for_pretty_print(row_labels=row_labels_dp_table, col_labels=col_labels)
    K.solve()

    alloc = K.get_output()[1][-1][-1]
    reward = K.get_output()[0][-1][-1]
    rewards_aggregated.append(reward)
    print(f"best allocation: {alloc}, total budget: {sum(alloc)}")
    print(f"reward: {reward}")
    set_budgets_env(alloc)
    sim_obj_2 = environment.replicate_last_day(N_user,
                                               reference_price,
                                               bool_n_noise,
                                               bool_n_noise)  # object with all the day info
    profit1, profit2, profit3, daily_profit = sim_obj_2["profit"]
    print(
        f"test allocation on env:\n\t || total:{daily_profit:.2f}€ || u1:{profit1:.2f}€ || u2:{profit2:.2f}€ || u3:{profit3:.2f}€")

    usr_context1 = filtered_users(users, 1, 1)
    usr_context2 = filtered_users(users, 1, 0)
    split_usr1 = usr_context1[0] == usr_context2[0]
    context_profit = 0
    sim_ctx1 = environment.contextualized_day(usr_context1,
                                              1,
                                              N_user,
                                              reference_price,
                                              bool_n_noise,
                                              bool_n_noise,
                                              split_usr1=True)
    p_cmp_1, p_cmp_2, p_cmp_3, p_cmp_4, p_cmp_5, tot_camp = sim_ctx1["profit_campaign"]
    print(f"CTX1: {tot_camp}")
    context_profit += tot_camp
    sim_ctx2 = environment.contextualized_day(usr_context2,
                                              1,
                                              N_user,
                                              reference_price,
                                              bool_n_noise,
                                              bool_n_noise,
                                              split_usr1=True)
    p_cmp_1, p_cmp_2, p_cmp_3, p_cmp_4, p_cmp_5, tot_camp = sim_ctx2["profit_campaign"]
    print(f"CTX2: {tot_camp}")
    context_profit += tot_camp

    print(f" test 0 == [{int(context_profit - daily_profit)}]")
"""
