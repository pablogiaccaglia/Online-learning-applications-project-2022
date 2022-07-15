import numpy as np
from Knapsack import Knapsack
from Part3.CombWrapper import CombWrapper
from Part3.GPTS_Learner import GPTS_Learner
from simulation.Environment import Environment
import matplotlib.pyplot as plt
import progressbar
from copy import deepcopy

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
days = 400
N_user = 350  # reference for what alpha = 1 refers to
reference_price = 3.0
daily_budget = 50 * 5
n_arms = 35
arm_distance = 5
step_k = 5
n_budget = int(daily_budget / step_k)
environment = Environment()

if daily_budget < arm_distance * n_arms:
    raise ValueError("Invalid Configuration for daily budget")

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

# ******* Context initialization ********
context_gen_days = 40
interval = 2  # how often to run context gen.
rewards_clairvoyant = []
base_learner_rewards = []
ctx_learner_rewards = []
rewards_knapsack_agg = []
budgets_array = np.array([
    [1, 1, 1, 1, 1],  # u 11
    [1, 1, 1, 1, 1],  # u 12
    [1, 1, 1, 1, 1],  # u 2
    [1, 1, 1, 1, 1]  # u3
]) * 50

context_on = False
target_feature = [False, False]  # start fully aggregated
stop_context = False
context_initialized = False
target_feature_i = 0

ctx_algorithm = GPTS_Learner
# ctx_algorithm = GTS_Learner

img, axss = plt.subplots(nrows=4, ncols=5, figsize=(13, 6))  # alpha plots
axs = axss.flatten()

img2, axss2 = plt.subplots(nrows=2, ncols=2, figsize=(13, 6))  # reward plots
axs2 = axss2.flatten()
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


def set_budgets_arm_env(_s_arm):
    for i, b in enumerate(_s_arm):
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


def budget_array_from_k_alloc_4(_alloc, flatten=False):
    """ map the super arm result with blocks of budget for every possible context participant """
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

""" Istantiate the new learner using a copy of the old trained comb lerner"""
"""def instantiate_new_learner(comb_learner, _contexts, _new_contexts, _ctx_algorithm, _n_campaigns_ctx, _n_arms,
                            _daily_budget):
    old_learners = deepcopy(comb_learner.learners)  # get list of learners
    old_learners = np.array_split(old_learners, len(_contexts))  # split in groups
    new_learners = []
    # n chunks = n new context

    for new_ctx in _new_contexts:  # number of times a chunk need to be allocated
        for j, old_ctx in enumerate(_contexts):  # j will be the index of the chunk to add
            test = np.sum(np.array(new_ctx) * np.array(old_ctx))  # matching condition between contexts
            if test > 0:
                new_learners.append(deepcopy(old_learners[j]))  # add chunk in order
                break
    learner = CombWrapper(_ctx_algorithm, _n_campaigns_ctx, _n_arms, _daily_budget, arm_distance)
    learner.learners = np.array(new_learners).flatten()

    return learner"""


def instantiate_new_learner(comb_learner, _contexts, _new_contexts, _ctx_algorithm, _n_campaigns_ctx, _n_arms,
                            _daily_budget):
    learner = CombWrapper(ctx_algorithm, _n_campaigns_ctx, _n_arms, _daily_budget, arm_distance)

    return learner


def set_budgets_env(knapsack_alloc):
    for i, b in enumerate(knapsack_alloc[1:]):
        environment.set_campaign_budget(i, b)


def reward_plot(active=True):
    if active:
        _axs = axs2  # target axs
        if day % 10 == 0:
            _axs[0].cla()
        if day % 2 == 0:
            x = available_budget
            d = np.linspace(0, len(rewards_clairvoyant), len(rewards_clairvoyant))
            _axs[0].set_xlabel("days")
            _axs[0].set_ylabel("reward")
            _axs[0].plot(d, rewards_clairvoyant)
            _axs[0].plot(d, rewards_clairvoyant)
            _axs[0].plot(d, rewards_knapsack_agg)
            _axs[0].plot(d, base_learner_rewards)
            plt.pause(0.1)


def alpha_plot(comb_learner, active=True):
    if active:
        ordered_rewards = []

        for offset in [0, 1, 2, 3]:  # order rewards from groups by users to groups by campaign
            for i_campaign in range(5):
                ordered_rewards.append(rewards[offset + 4 * i_campaign])
        ordered_rewards = np.array_split(ordered_rewards, 4)
        agg_ordered_rewards = []

        for _mask in contexts:  # aggregate knapsack rewards according to actual context
            tmp = np.array(ordered_rewards[0]) * 0
            bool_discount_cost = False
            for i_bit, bit in enumerate(_mask):
                if bit == 1:
                    tmp += ordered_rewards[i_bit]
            agg_ordered_rewards.append(tmp)

        _axs = axs  # target axs
        rewards_plot = np.array(agg_ordered_rewards).reshape(-1, len(available_budget))
        if day % 20 == 0:
            for i, rw in enumerate(rewards_plot):
                _axs[i].cla()
        if day % 4 == 0:
            x = available_budget
            x2 = comb_learner.arms
            for i, rw in enumerate(rewards_plot):
                # _axs[i].set_xlabel("budget")
                # _axs[i].set_ylabel("profit")
                _axs[i].plot(x, rw)
                # axs[i].plot(x2, comb_learner.last_knapsack_reward[i])
                mean, std = comb_learner.get_gp_data()
                # print(std[0])
                # print(mean[0][0])
                _axs[i].plot(x2, mean[i])
                _axs[i].fill_between(
                    np.array(x2).ravel(),
                    mean[i] - 1.96 * std[i],
                    mean[i] + 1.96 * std[i],
                    alpha=0.5,
                    label=r"95% confidence interval",
                )
            plt.pause(0.3)


def knapsack_aggregated():
    # aggregated knapsack   --------------------------------------------
    row_label_rewards, row_labels_dp_table, col_labels = table_metadata(len(products), 1, available_budget)

    K = Knapsack(rewards=rewards, budgets=np.array(available_budget))
    K.init_for_pretty_print(row_labels=row_labels_dp_table, col_labels=col_labels)
    K.solve()
    arg_max = np.argmax(K.get_output()[0][-1])
    alloc = K.get_output()[1][-1][arg_max]
    reward = K.get_output()[0][-1][arg_max]
    # rewards_knapsack_agg.append(reward)

    # set knapsack solution on env
    k_alloc = alloc[1:]
    # print(f"\nk: {k_alloc}")
    set_budgets_arm_env(k_alloc)
    # test result on env
    sim_obj_2 = environment.replicate_last_day(N_user,
                                               reference_price,
                                               bool_n_noise,
                                               bool_n_noise)

    rewards_knapsack_agg.append(sim_obj_2["profit_campaign"][-1] - np.sum(k_alloc))
    # -----------------------------------------------------------------


def knapsack_disaggregated():
    # disaggregated knapsack   --------------------------------------------
    row_label_rewards, row_labels_dp_table, col_labels = table_metadata(len(products), len(users), available_budget)
    K = Knapsack(rewards=rewards, budgets=np.array(available_budget))
    K.init_for_pretty_print(row_labels=row_labels_dp_table, col_labels=col_labels)
    K.solve()

    arg_max = np.argmax(K.get_output()[0][-1])
    alloc = K.get_output()[1][-1][arg_max]
    reward = K.get_output()[0][-1][arg_max]

    b_knap = budget_array_from_k_alloc_4(alloc)  # budgets vector for contextualized env
    profit_blocks = environment.get_context_building_blocks(budgets_array=b_knap,
                                                            n_users=N_user,
                                                            reference_price=reference_price)  # test clairvoyant on env
    gross_profit_k_env = np.sum(assemble_profit(profit_blocks, contexts, p_users))
    net_profit_k_env = gross_profit_k_env - np.sum(alloc)
    rewards_clairvoyant.append(net_profit_k_env)
    # -------------------------------------------------------------------------


# initialize base learner target and shape
contexts = context_masks(split_family=target_feature[0], split_student=target_feature[1])
size_ctx = len(contexts)
start_offset = 0
n_campaigns_ctx = 5 * size_ctx
base_learner = CombWrapper(ctx_algorithm, n_campaigns_ctx, n_arms, daily_budget, arm_distance)
super_arm = base_learner.pull_super_arm()
last_superarm = super_arm

boost_start = True
boost_discount = 0.4  # boost discount wr to the highest reward
boost_bias = daily_budget / 20  # ensure a positive reward when all pull 0
# *********************************************************************

for day in progressbar.progressbar(range(days)):
    users, products, campaigns, allocated_budget, prob_users = environment.get_core_entities()
    sim_obj = environment.play_one_day(N_user, reference_price, daily_budget, step_k, bool_alpha_noise,
                                       bool_n_noise)  # object with all the day info
    # aggregated knapsack   --------------------------------------------
    rewards, available_budget = sim_obj["reward_k_agg"]
    knapsack_aggregated()
    # -----------------------------------------------------------------
    # disaggregated knapsack   --------------------------------------------
    rewards, available_budget = sim_obj["reward_k_4"]
    knapsack_disaggregated()
    # ----------------------------------------------------------------------

    if day == 40 or day == 100:
        if target_feature_i <= 1:
            context_on = True
            last_superarm = super_arm  # save arm of possible old learner
            target_feature[target_feature_i] = True  # split for next feature
            target_feature_i += 1
            start_day = day
            print(f"context ON - day {day} bool1: {target_feature[0]} bool2: {target_feature[1]}")

    if context_on:
        # generate contexts list of context masks
        new_contexts = context_masks(split_family=target_feature[0], split_student=target_feature[1])
        size_ctx = len(new_contexts)

        if not context_initialized:
            # initialize candidate learner
            n_campaigns_ctx = 5 * size_ctx
            old_learner = base_learner
            ctx_learner = instantiate_new_learner(base_learner,
                                                  contexts,
                                                  new_contexts,
                                                  ctx_algorithm,
                                                  n_campaigns_ctx,
                                                  n_arms,
                                                  daily_budget)

            super_arm = ctx_learner.pull_super_arm()  # pull arm for new learner
            contexts = new_contexts
            context_initialized = True

        # generate budget matrix from pulled superarm
        budgets_array = budget_array_from_superarm(super_arm, contexts, p_users)
        # get gross profits from env
        profit_blocks = environment.get_context_building_blocks(budgets_array=budgets_array,
                                                                n_users=N_user,
                                                                reference_price=reference_price)

        learner_reward = assemble_profit(profit_blocks, contexts, p_users)  # gross profit
        learner_reward = learner_reward.flatten() - np.array(super_arm)  # net profit

        profit_list = list(learner_reward.flatten())
        if boost_start and day < 44:
            for i, s_arm in enumerate(super_arm):
                if s_arm == 0:
                    # if a learner is pulling 0 give an high reward in an higher arm
                    back_offset = np.random.randint(1, 4)
                    forced_arm = np.sort(super_arm, axis=None)[-back_offset]  # take random high arm value
                    profit_list[i] = np.max(profit_list) * boost_discount + boost_bias
                    super_arm[i] = forced_arm

        ctx_learner.update_observations(super_arm, learner_reward)

        # collect net profit of learner
        ctx_learner_rewards.append(np.sum(learner_reward))
        base_learner_rewards.append(np.sum(learner_reward))

        # pull super arm for tomorrow
        super_arm = ctx_learner.pull_super_arm()

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

        reward_plot(active=True)
        alpha_plot(ctx_learner)
    else:
        budgets_array = budget_array_from_superarm(super_arm, contexts, p_users)
        # get profits from env (need super arm)
        profit_blocks = environment.get_context_building_blocks(budgets_array=budgets_array,
                                                                n_users=N_user,
                                                                reference_price=reference_price)
        learner_reward = assemble_profit(profit_blocks, contexts, p_users)
        # TRY give penalty ( seems better without penalty)

        learner_reward = learner_reward.flatten() - np.array(super_arm)
        net_profit_learner = np.sum(learner_reward)

        base_learner.update_observations(super_arm, learner_reward.flatten())
        base_learner_rewards.append(net_profit_learner)

        # pull super arm for tomorrow
        super_arm = base_learner.pull_super_arm()

        reward_plot(active=True)
        alpha_plot(base_learner)
plt.show()

"""# ********* statistical measures *********************
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
