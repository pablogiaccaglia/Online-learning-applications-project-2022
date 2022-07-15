import numpy as np

from Environment import Environment
from SimulationHandler import SimulationHandler

if __name__ == '__main__':
    """ @@@@ simulations SETUP @@@@ """
    experiments = 2
    days = 2
    N_user = 300  # reference for what alpha = 1 refers to
    reference_price = 4.0
    daily_budget = 50 * 5
    step_k = 2
    n_arms = int(np.ceil(np.power(days * np.log(days), 0.25))) + 1

    bool_alpha_noise = False
    bool_n_noise = False
    printBasicDebug = False
    printKnapsackInfo = True

    simulationHandler = SimulationHandler(environmentConstructor = Environment,
                                          learners = [],
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
                                          clairvoyant_type = 'both'
                                          )

    simulationHandler.run_simulation()
