import sys
from numbers import Number
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd

np.set_printoptions(threshold=sys.maxsize)


class Knapsack:

    def __init__(self, rewards: np.ndarray = None, budgets: np.ndarray = None) -> None:
        self.reset(rewards=rewards, budgets=budgets)
        self.NINF = -100

    def solve(self) -> None:

        if not (isinstance(self.rewards, np.ndarray) and isinstance(self.budgets, np.ndarray)):
            raise Exception("You need to properly initialize the problem by calling reset method")

        self.optimized = False
        # cycle for each row of the dp_table
        for row in range(1, self.rows):
            # cycle for each row of the dp_table

            # now cycle through each column until the current one (included, that's why there is a +1)
            for column in range(self.columns):

                # initialize max value for each cell to minus infinity
                max_value = self.NINF

                """set table entry to NINF in case of infeasible budget for the new campaign considered and
                previous combination of campaigns considered.

                In particular, if the reward of the previous combination equals to 0 and the new reward equals
                to NINF, or the reward of the previous combination equals to NINF and the new reward equals to 0,
                or both reward of the previous cominbation and the new reward equal to NINF
                then set the new entry of the dp table to NINF

                """
                if (self.rewards[row - 1][column] < 0 and self.dp_table[row - 1][column] == 0) \
                        or (self.rewards[row - 1][column] == 0 and self.dp_table[row - 1][column] < 0) or \
                        (self.rewards[row - 1][column] < 0 and self.dp_table[row - 1][column] < 0):
                    self.dp_table[row][column] = self.NINF
                    continue

                for index in range(column + 1):
                    # the current value is the sum of the subcampaign reward associated to [index] and the value of
                    # the dp table associated to [previous row][column-index]
                    # this way the sum is always equal to the budget expressed by the column

                    if self.dp_table[row - 1][column - index] >= 0 and self.rewards[row - 1][index] >= 0:
                        current_value = self.rewards[row - 1][index] + self.dp_table[row - 1][column - index]
                    else:
                        if self.rewards[row - 1][index] >= 0:
                            current_value = self.rewards[row - 1][index]
                        else:
                            current_value = self.dp_table[row - 1][column]

                        # update max value
                    if current_value > max_value:
                        max_value = current_value
                        allocation = np.copy(self.allocations[row - 1][column - index])
                        allocation[row] = self.budgets[index]

                # update max value in the dp table
                if row > 0:
                    self.allocations[row][column] = allocation

                self.dp_table[row][column] = max_value

        self.optimized = True

        if self.toRestore:
            # get rid of first column of zeros
            self.dp_table = self.dp_table[:, 1:]

            self.budgets = self.budgets[1:]

            self.rewards = self.rewards[: 1:]

            self.allocations = self.allocations[:, 1:, :]

            self.columns -= 1

    def init_for_pretty_print(self, row_labels, col_labels) -> None:
        self.row_labels = row_labels
        self.column_labels = col_labels
        self.initialized_pretty_print = True

    def pretty_print_output(self, print_last_row_only=False):

        def sequence_of_ints_strings(start: int, end: int) -> str:
            sequence = ""
            for i in range(start, end):
                sequence = sequence + str(i) + ", "

            sequence = sequence + str(end)

            return sequence

        def format_allocation(start_campaign: int,
                              end_campaign: int,
                              allocs: np.ndarray,
                              row: int, col: int, spaces: int):
            alloc_formatted = ""
            for i in range(start_campaign, end_campaign):
                alloc_formatted = alloc_formatted + "Campaign " + str(i) + "->" + str(allocs[row][col][i]) + "|"
                alloc_formatted = alloc_formatted + "\n|" + " " * spaces

            alloc_formatted = alloc_formatted + "Campaign " + str(end_campaign) + "->" + str(
                allocs[row][col][end_campaign]) + "|"

            return alloc_formatted

        if not self.optimized:
            raise Exception("Run optimization first!")

        formatted_output = ""

        start = self.rows - 1 if print_last_row_only else 0
        for row in range(start, self.rows):
            formatted_output = formatted_output + "\nReward for campaign(s) " + sequence_of_ints_strings(1, row) + ":\n"
            for column in range(0, self.columns):
                budget_str = "\n|Budget: " + str(self.budgets[column])
                rew_str = "\t|Reward: " + str(self.dp_table[row][column]) + "|"
                alloc_str = "\n|Allocations: " + format_allocation(start_campaign=1,
                                                                   end_campaign=row,
                                                                   allocs=self.allocations,
                                                                   row=row,
                                                                   col=column,
                                                                   spaces=len("Allocations: "))

                formatted_output = formatted_output + budget_str + rew_str + alloc_str
                formatted_output = formatted_output + "\n\t" + "--" * 5 + "\n"

        print("*" * 30 + " Knapsack output " + "*" * 35)
        print(formatted_output)

    def get_output(self, dp_as_dataframe=False) -> Tuple[Union[pd.DataFrame, np.ndarray], np.ndarray]:
        if not self.optimized:
            raise Exception("Run optimization first!")

        if dp_as_dataframe:
            dp_table_dataframe = pd.DataFrame(self.dp_table, columns=self.column_labels, index=self.row_labels)
            return dp_table_dataframe, self.allocations
        else:
            return self.dp_table, self.allocations

    def pretty_print_dp_table(self, multiplier: Number = None):
        if not self.optimized:
            raise Exception("Run optimization first!")

        if not self.initialized_pretty_print:
            raise Exception("You have to call init_for_pretty_print method first")

        dp_table = self.dp_table

        if multiplier:
            dp_table = self.dp_table.copy() * multiplier

        df = pd.DataFrame(dp_table, columns=self.column_labels, index=self.row_labels)

        pd.options.display.width = 0

        print("*" * 8 + " Combinatorial optimization table " + "*" * 8 + "\n" + str(df))

    def reset(self, rewards: np.ndarray, budgets: np.ndarray) -> None:

        if not (isinstance(rewards, np.ndarray) and isinstance(budgets, np.ndarray)):
            return

        # initialize budgets vector
        self.budgets = budgets.copy()

        # initialize rewards matrix
        self.rewards = rewards.copy()

        # we need to ensure that the first value for the budget is 0, because this is needed for the algorithm to handle
        # the case in which we allocate no budget to a certain campaign. As a consequence we need to modify the rewards
        # matrix too.

        # boolean flag to keep track of modifications of input data, see below.
        self.toRestore = False

        if self.budgets[0] != 0:
            self.budgets = np.insert(self.budgets, 0, 0)
            self.rewards = np.append(np.zeros((self.rewards.shape[0], 1), dtype=self.rewards.dtype), self.rewards, 1)

            # we keep track of the changes to restore the original shape when returning results.
            self.toRestore = True

        # initialize allocations tensor
        self.allocations = np.zeros((self.rewards.shape[0] + 1, self.rewards.shape[1], self.rewards.shape[0] + 1),
                                    dtype=int)

        # initialize dynamic programming table, dimensions: (subcampaigns + 1, budgets)
        self.dp_table = np.zeros((len(self.rewards) + 1, len(self.budgets)), dtype=float)

        self.rows = self.dp_table.shape[0]
        self.columns = self.dp_table.shape[1]

        self.row_labels = []  # dp table's row names
        self.column_labels = []  # dp table's column names

        # flags to handle pretty print output functions calls
        self.optimized = False
        self.initialized_pretty_print = False