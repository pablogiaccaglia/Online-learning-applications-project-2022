from learners.CombWrapper import CombWrapper
from GPTS_Learner import GPTS_Learner
from SimulationHandler import SimulationHandler
from simulations.Environment import Environment
import numpy as np
if __name__ == '__main__':
    """ @@@@ simulations SETUP @@@@ """

    experiments = 2
    days = 80
    N_user = 300 # reference for what alpha = 1 refers to
    reference_price = 4.0
    daily_budget = 500
    step_k = 2
    n_arms = int(np.ceil(np.power(days * np.log(days), 0.25))) + 1

    bool_alpha_noise = True
    bool_n_noise = False
    printBasicDebug = False
    printKnapsackInfo = True
    runAggregated = False  # mutual exclusive with run disaggregated

    """ Change here the wrapper for the core bandit algorithm """
    # comb_learner = CombWrapper(GTS_Learner, 5, n_arms, daily_budget, is_ucb = False, is_gaussian = True)
    comb_learner = CombWrapper(GPTS_Learner, 5, n_arms, daily_budget, is_ucb = False, is_gaussian = True)
    learners = [comb_learner]
    simulationHandler = SimulationHandler(environmentConstructor = Environment,
                                          learners = learners,
                                          experiments = experiments,
                                          days = days,
                                          reference_price = reference_price,
                                          daily_budget = daily_budget,
                                          n_users = N_user,
                                          n_arms = n_arms,
                                          bool_alpha_noise = bool_alpha_noise,
                                          bool_n_noise = bool_n_noise,
                                          print_basic_debug = printBasicDebug,
                                          print_knapsack_info = printKnapsackInfo,
                                          step_k = step_k,
                                          is_unknown_graph = True
                                          )

    simulationHandler.run_simulation()


