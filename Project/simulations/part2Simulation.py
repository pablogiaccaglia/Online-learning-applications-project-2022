import numpy as np

from simulations.Environment import Environment
from simulations.SimulationHandler import SimulationHandler

if __name__ == '__main__':
    """ @@@@ simulations SETUP @@@@ """
    experiments = 100
    days = 80
    N_user = 350  # reference for what alpha = 1 refers to
    reference_price = 4.0
    daily_budget = 50 * 6
    step_k = 5
    n_arms = 3 * int(np.ceil(np.power(days * np.log(days), 0.25))) + 1

    bool_alpha_noise = False
    bool_n_noise = False
    printBasicDebug = False
    printKnapsackInfo = True

    simulationHandler = SimulationHandler(environmentConstructor=Environment,
                                          learners=[],
                                          experiments=experiments,
                                          days=days,
                                          reference_price=reference_price,
                                          daily_budget=daily_budget,
                                          campaigns = 5,
                                          n_users=N_user,
                                          n_arms=n_arms,
                                          bool_alpha_noise=bool_alpha_noise,
                                          bool_n_noise=bool_n_noise,
                                          print_basic_debug=printBasicDebug,
                                          print_knapsack_info=printKnapsackInfo,
                                          step_k=step_k,
                                          clairvoyant_type='both',
                                          simulation_name = 'Part2Simulation',
                                          )

    simulationHandler.run_simulation()
