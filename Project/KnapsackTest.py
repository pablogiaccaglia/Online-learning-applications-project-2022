import numpy as np
import pytest

from Knapsack import Knapsack


class TestKnapsack:

    def setup(self):
        self.rew = np.empty((0, 0))
        self.budgets = np.empty((0, 0))
        self.row_labels = []
        self.col_labels = []
        self.K = Knapsack()
        self.NINF = -100

    def testBaseKnp(self) -> None:
        self.rew = np.empty(shape = (5, 8), dtype = np.int32)

        self.rew[0] = np.array([self.NINF, 90, 100, 105, 110, self.NINF, self.NINF, self.NINF])
        self.rew[1] = np.array([0, 82, 90, 92, self.NINF, self.NINF, self.NINF, self.NINF])
        self.rew[2] = np.array([0, 80, 83, 85, 86, self.NINF, self.NINF, self.NINF])
        self.rew[3] = np.array([self.NINF, 90, 110, 115, 118, 120, self.NINF, self.NINF])
        self.rew[4] = np.array([self.NINF, 111, 130, 138, 142, 148, 155, self.NINF])
        self.budgets = np.array([0, 10, 20, 30, 40, 50, 60, 70])

        self.row_labels = ['0', '+C1', '+C2', '+C3', '+C4', '+C5']
        self.col_labels = [str(budget) for budget in self.budgets]

        self.K.reset(rewards = self.rew, budgets = self.budgets)
        self.K.init_for_pretty_print(row_labels = self.row_labels, col_labels = self.col_labels)
        self.K.solve()

        dp_table, _ = self.K.get_output()

        target_shape = (6, 8)
        if dp_table.shape != target_shape:
            raise Exception("Test failed")

        dp_table_true = np.empty(shape = target_shape, dtype = np.float16)

        dp_table_true[0] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        dp_table_true[1] = np.array([-100.0, 90.0, 100.0, 105.0, 110.0, -100.0, -100.0, -100.0])
        dp_table_true[2] = np.array([-100.0, 90.0, 172.0, 182.0, 190.0, -100.0, -100.0, -100.0])
        dp_table_true[3] = np.array([-100.0, 90.0, 172.0, 252.0, 262.0, -100.0, -100.0, -100.0])
        dp_table_true[4] = np.array([-100.0, 90.0, 180.0, 262.0, 342.0, 362.0, -100.0, -100.0])
        dp_table_true[5] = np.array([-100.0, 111.0, 201.0, 291.0, 373.0, 453.0, 473.0, -100.0])

        if not np.equal(dp_table, dp_table_true).all():
            self.K.pretty_print_dp_table()
            raise Exception("**" * 5 + " Test base failed " + "**" * 5)

    def testAll0s(self) -> None:
        self.rew = np.zeros(shape = (5, 8), dtype = np.int32)
        self.row_labels = ['0', '+C1', '+C2', '+C3', '+C4', '+C5']
        self.col_labels = [str(budget) for budget in self.budgets]
        self.budgets = np.array([0, 10, 20, 30, 40, 50, 60, 70])

        self.K.reset(rewards = self.rew, budgets = self.budgets)
        self.K.init_for_pretty_print(row_labels = self.row_labels, col_labels = self.col_labels)
        self.K.solve()

        dp_table, _ = self.K.get_output()

        target_shape = (6, 8)
        if dp_table.shape != target_shape:
            raise Exception("Test failed")

        dp_table_true = np.zeros(shape = target_shape, dtype = np.float16)
        if not np.equal(dp_table, dp_table_true).all():
            self.K.pretty_print_dp_table()
            raise Exception("**" * 5 + " Test all zeros failed " + "**" * 5)

    def testAllNegative(self) -> None:
        self.rew = np.ones(shape = (5, 8), dtype = np.int32) * self.NINF
        self.row_labels = ['0', '+C1', '+C2', '+C3', '+C4', '+C5']
        self.col_labels = [str(budget) for budget in self.budgets]
        self.budgets = np.array([0, 10, 20, 30, 40, 50, 60, 70])

        self.K.reset(rewards = self.rew, budgets = self.budgets)
        self.K.init_for_pretty_print(row_labels = self.row_labels, col_labels = self.col_labels)
        self.K.solve()

        dp_table, _ = self.K.get_output()

        target_shape = (6, 8)
        if dp_table.shape != target_shape:
            print(dp_table.shape)
            raise Exception("Test failed")

        dp_table_true = np.ones(shape = target_shape, dtype = np.float16) * self.NINF
        dp_table_true[0] *= 0.0
        if not np.equal(dp_table, dp_table_true).all():
            self.K.pretty_print_dp_table()
            raise Exception("**" * 5 + " Test all negative failed " + "**" * 5)

    def testAlternatingNegs(self) -> None:
        self.rew = np.empty(shape = (2, 8), dtype = np.int32)

        self.rew[0] = np.array([self.NINF, 20, self.NINF, 40, self.NINF, 60, self.NINF, 100])
        self.rew[1] = np.array([40, self.NINF, 50, self.NINF, 60, self.NINF, 70, self.NINF])
        self.budgets = np.array([0, 10, 20, 30, 40, 50, 60, 70])

        self.row_labels = ['0', '+C1', '+C2']
        self.col_labels = [str(budget) for budget in self.budgets]

        self.K.reset(rewards = self.rew, budgets = self.budgets)
        self.K.init_for_pretty_print(row_labels = self.row_labels, col_labels = self.col_labels)
        self.K.solve()

        dp_table, _ = self.K.get_output()

        target_shape = (3, 8)
        if dp_table.shape != target_shape:
            raise Exception("Test failed")

        dp_table_true = np.empty(shape = target_shape, dtype = np.float16)

        dp_table_true[0] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        dp_table_true[1] = np.array([self.NINF, 20, self.NINF, 40, self.NINF, 60, self.NINF, 100])
        dp_table_true[2] = np.array([40, 60, 50, 80, 60, 100, 70, 140])

        if not np.equal(dp_table, dp_table_true).all():
            self.K.pretty_print_dp_table()
            raise Exception("**" * 5 + " Test alternating neg failed " + "**" * 5)

    def testAlternatingNegsAndZeros(self) -> None:

        self.rew = np.empty(shape = (2, 8), dtype = np.int32)

        self.rew[0] = np.array([self.NINF, 20, self.NINF, 40, self.NINF, 60, self.NINF, 100])
        self.rew[1] = np.array([40, 0, 50, 0, 60, 0, 70, 0])
        self.budgets = np.array([0, 10, 20, 30, 40, 50, 60, 70])

        self.row_labels = ['0', '+C1', '+C2']
        self.col_labels = [str(budget) for budget in self.budgets]

        self.K.reset(rewards = self.rew, budgets = self.budgets)
        self.K.init_for_pretty_print(row_labels = self.row_labels, col_labels = self.col_labels)
        self.K.solve()

        dp_table, _ = self.K.get_output()

        target_shape = (3, 8)
        if dp_table.shape != target_shape:
            raise Exception("Test failed")

        dp_table_true = np.empty(shape = target_shape, dtype = np.float16)

        dp_table_true[0] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        dp_table_true[1] = np.array([self.NINF, 20, self.NINF, 40, self.NINF, 60, self.NINF, 100])
        dp_table_true[2] = np.array([40, 60, 50, 80, 60, 100, 70, 140])

        if not np.equal(dp_table, dp_table_true).all():
            self.K.pretty_print_dp_table()
            raise Exception("**" * 5 + " Test alternating neg failed " + "**" * 5)

    def testSomeNegs(self) -> None:

        self.rew = np.empty(shape = (2, 8), dtype = np.int32)

        self.rew[0] = np.array([self.NINF, 20, 30, 40, 50, 60, 70, 80])
        self.rew[1] = np.array([self.NINF, self.NINF, self.NINF, self.NINF, 60, 70, 80, self.NINF])

        self.budgets = np.array([0, 10, 20, 30, 40, 50, 60, 70])

        self.row_labels = ['0', '+C1', '+C2']
        self.col_labels = [str(budget) for budget in self.budgets]

        self.K.reset(rewards = self.rew, budgets = self.budgets)
        self.K.init_for_pretty_print(row_labels = self.row_labels, col_labels = self.col_labels)
        self.K.solve()

        dp_table, _ = self.K.get_output()

        target_shape = (3, 8)
        if dp_table.shape != target_shape:
            raise Exception("Test failed")

        dp_table_true = np.zeros(shape = target_shape, dtype = np.float16)

        dp_table_true[0] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        dp_table_true[1] = np.array([self.NINF, 20, 30, 40, 50, 60, 70, 80])
        dp_table_true[2] = np.array([self.NINF, 20, 30, 40, 60, 80, 90, 100])

        if not np.equal(dp_table, dp_table_true).all():
            self.K.pretty_print_dp_table()
            raise Exception("**" * 5 + " Test some negs failed " + "**" * 5)

    def testAll(self) -> None:
        self.setup()
        self.testBaseKnp()
        self.testAll0s()
        self.testAllNegative()
        self.testAlternatingNegs()
        self.testAlternatingNegsAndZeros()
        self.testSomeNegs()


"""KT = TestKnapsack()
KT.testAll()
"""