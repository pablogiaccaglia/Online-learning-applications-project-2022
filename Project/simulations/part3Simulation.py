import numpy as np

from GPUCB1_Learner import GPUCB1_Learner
from learners.GPTS_Learner import GPTS_Learner
from learners.CombWrapper import CombWrapper
from simulations.Environment import Environment
from simulations.SimulationHandler import SimulationHandler
from entities.Utils import BanditNames
if __name__ == '__main__':
    """ @@@@ simulations SETUP @@@@ """
    experiments = 2
    days = 3
    N_user = 300  # reference for what alpha = 1 refers to
    reference_price = 4.0
    daily_budget = 50 * 5
    step_k = 5
    n_arms = int(np.ceil(np.power(days * np.log(days), 0.25))) + 1

    bool_alpha_noise = True
    bool_n_noise = False
    printBasicDebug = False
    printKnapsackInfo = True

    boost_start = True
    boost_discount = 0.5  # boost discount wr to the highest reward
    boost_bias = daily_budget / 5  # ensure a positive reward when all pull 0

    """ @@@@ ---------------- @@@@ """

    gpts_learner = CombWrapper(GPTS_Learner, 5, n_arms, daily_budget,
                               is_ucb=False,
                               is_gaussian=True)

    gpucb1_learner = CombWrapper(GPUCB1_Learner,
                                 5,
                                 n_arms,
                                 daily_budget,
                                 is_ucb = True,
                                 is_gaussian = True)

    learners = [gpts_learner, gpucb1_learner]

    simulationHandler = SimulationHandler(environmentConstructor=Environment,
                                          learners=learners,
                                          experiments=experiments,
                                          days=days,
                                          reference_price=reference_price,
                                          daily_budget=daily_budget,
                                          n_users=N_user,
                                          n_arms=n_arms,
                                          bool_alpha_noise=bool_alpha_noise,
                                          bool_n_noise=bool_n_noise,
                                          print_basic_debug=printBasicDebug,
                                          print_knapsack_info=printKnapsackInfo,
                                          step_k=step_k,
                                          clairvoyant_type='aggregated',
                                          boost_start=boost_start,
                                          boost_bias=boost_bias,
                                          boost_discount=boost_discount,
                                          plot_regressor_progress= BanditNames.GPTS_Learner.name
                                          )

    simulationHandler.run_simulation()
