class Campaign:
    def __init__(self,
                 id,
                 allocated_budget,
                 alpha_i_max):
        self.id = id
        self.allocated_budget = allocated_budget
        self.alpha_i_max = alpha_i_max

    def change_budget(self, new_budget):
        self.allocated_budget = new_budget

    def get_alpha_i(self, user_alpha_function):
        """ Function used to compute the reaction of an user to a campaign instance, the function
            takes as input the alpha function of the user and compute it based on the allocated balance of the
            campaign upper-bounding it to alpha_i_max """
        return user_alpha_function(self.allocated_budget).clip(0.0) * self.alpha_i_max