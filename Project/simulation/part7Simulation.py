import numpy as np
import pandas as pd
from Knapsack import Knapsack
from Part3.CombWrapper import CombWrapper
from Part3.GPTS_Learner import GPTS_Learner
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

days = 100
N_user = 200  # reference for what alpha = 1 refers to
reference_price = 4.0
daily_budget = 50 * 5
n_arms = 15
step_k = 5
context_gen_days = 25
interval = 40   # how often to run context gen.
n_budget = int(daily_budget / step_k)
environment = Environment()

bool_alpha_noise = True
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


def filtered_users(users, split_family=False, split_student=False, debug=False):
    final_list = users.copy()
    try:
        if family == 1:
            final_list.remove(users[1])
            final_list.remove(users[2])
        if student == 1:
            final_list.remove(users[2])
        if family == 0:
            final_list.remove(users[0])
        if student == 0:
            final_list.remove(users[1])
    except ValueError:
        pass  # do nothing
    if debug:
        for i in final_list:
            for inx, usr in enumerate(users):
                if i == usr:
                    print(f"User {inx + 1}")
        print("\n")

    return final_list


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


def assemble_profit(_profit_blocks, _contexts, _p_users):
    assembled_profit = []
    for mask in _contexts:
        tmp_array = np.zeros(5)
        for i_user, bit in enumerate(mask):
            if bit == 1:
                tmp_array += _profit_blocks[i_user] * _p_users[i_user]
        assembled_profit.append(tmp_array)
    return np.array(assembled_profit).flatten()


def budget_array_from_superarm(_super_arm, _contexts):
    """ map the super arm result with blocks of budget for every possible context participant """
    if len(_super_arm) / len(_contexts) != 5.0:
        raise ValueError("Super arm not compatible with context")
    budgets = np.array(_super_arm).reshape((len(_contexts), 5))
    result = np.zeros((4, 5))
    for i, ctx in enumerate(_contexts):
        permutation = np.array(ctx).reshape((len(ctx), 1))
        b = budgets[i].reshape((1, 5))
        result += permutation @ b

    return result

def set_budgets_env(knapsack_alloc):
    for i, b in enumerate(knapsack_alloc[1:]):
        environment.set_campaign_budget(i, b)


# ******* Context initialization ********
rewards_clairvoyant = []
base_learner_rewards = []
ctx_learner_rewards = []
budgets_array = np.array([
    [1, 1, 1, 1, 1],  # u 11
    [1, 1, 1, 1, 1],  # u 12
    [1, 1, 1, 1, 1],  # u 2
    [1, 1, 1, 1, 1]  # u3
]) * 40

context_on = False
target_feature = [False, False]  # start fully aggregated
stop_context = False
context_initialized = False

ctx_algorithm = GPTS_Learner

# initialize base learner target and shape
contexts = context_masks(split_family=target_feature[0], split_student=target_feature[1])
size_ctx = len(contexts)
n_campaigns_ctx = 5 * size_ctx
base_learner = CombWrapper(ctx_algorithm, n_campaigns_ctx, n_arms, daily_budget)
super_arm = base_learner.pull_super_arm()
# ***************************************

img, axss = plt.subplots(nrows=2, ncols=3, figsize=(13, 6))
axs = axss.flatten()

for day in progressbar.progressbar(range(days)):
    # print(f"Current context: {contexts}")
    users, products, campaigns, allocated_budget, prob_users = environment.get_core_entities()
    sim_obj = environment.play_one_day(N_user, reference_price, daily_budget, bool_alpha_noise,
                                       bool_n_noise)  # object with all the day info
    # disaggregated knapsack   --------------------------------------------
    rewards, available_budget = sim_obj["reward_k"]
    row_label_rewards, row_labels_dp_table, col_labels = table_metadata(len(products), len(users), available_budget)
    K = Knapsack(rewards=rewards, budgets=np.array(available_budget))
    K.init_for_pretty_print(row_labels=row_labels_dp_table, col_labels=col_labels)
    K.solve()

    arg_max = np.argmax(K.get_output()[0][-1])
    alloc = K.get_output()[1][-1][arg_max]
    reward = K.get_output()[0][-1][arg_max]
    rewards_clairvoyant.append(reward)

    # todo debug clairvoyant since gp is better probably misreport of reward
    # -------------------------------------------------------------------------
    if day == interval:
        context_on = True
    if context_on:
        contexts = context_masks(split_family=target_feature[0], split_student=target_feature[1])
        size_ctx = len(contexts)

        if not context_initialized:
            n_campaigns_ctx = 5 * size_ctx
            ctx_learner = CombWrapper(ctx_algorithm, n_campaigns_ctx, n_arms, daily_budget)
            super_arm = ctx_learner.pull_super_arm()
            context_initialized = True

        budgets_array = budget_array_from_superarm(super_arm, contexts)
        # get profits from env (need super arm)
        profit_blocks = environment.get_context_building_blocks(budgets_array=budgets_array,
                                                                n_users=N_user,
                                                                reference_price=reference_price)
        learner_reward = assemble_profit(profit_blocks, contexts, p_users)

        ctx_learner.update_observations(super_arm, learner_reward)
        ctx_learner_rewards.append(np.sum(learner_reward) - np.sum(super_arm))

        # pull super arm for tomorrow
        super_arm = ctx_learner.pull_super_arm()

        # print(f"blovk profit {profit_blocks}")
        # print(f"profit ready for bandit {ctx_profit}")
        # print(f"Env profits : {sim_obj['profit_campaign']}")

    else:
        budgets_array = budget_array_from_superarm(super_arm, contexts)
        # get profits from env (need super arm)
        profit_blocks = environment.get_context_building_blocks(budgets_array=budgets_array,
                                                                n_users=N_user,
                                                                reference_price=reference_price)
        learner_reward = assemble_profit(profit_blocks, contexts, p_users)
        base_learner.update_observations(super_arm, learner_reward)
        base_learner_rewards.append(np.sum(learner_reward) - np.sum(super_arm))
        ctx_learner_rewards.append(np.sum(learner_reward) - np.sum(super_arm))

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
        if context_on:
            axs[0].plot(d, ctx_learner_rewards)
        else:
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
