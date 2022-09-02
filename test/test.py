import unittest

import numpy as np

from mollifiers import smooth_step, smooth_function_exp, smooth_indicator, mollifier

class TestTest(unittest.TestCase):

    def test_operation(self):
        self.assertTrue(True)

class TestFunctionExp(unittest.TestCase):

    def test_zero(self):

        x = -5.0
        value = smooth_function_exp(x)
        self.assertEqual(value, 0.0)

        x = 0.0
        value = smooth_function_exp(x)
        self.assertEqual(value, 0.0)

    def test_positive(self):

        x = 1.0
        value = smooth_function_exp(x)
        self.assertAlmostEqual(value, np.exp(-1), delta=1e-14)

        x = 2.0
        value2 = smooth_function_exp(x)
        self.assertGreater(value2, value)

    def test_limit(self):
        
        x = np.inf
        limit_value = smooth_function_exp(x)
        self.assertEqual(limit_value, 1.0)

        x = -np.inf
        limit_value = smooth_function_exp(x)
        self.assertEqual(limit_value, 0.0)

    def test_vector(self):
        x = np.linspace(-1, 1, 20)
        x_to_zero = x <= 0

        y = smooth_function_exp(x)
        
        self.assertAlmostEqual(np.linalg.norm(y[x_to_zero]), 0.0, delta=1e-14)

        y_pos = y[~x_to_zero]
        for j in range(len(y_pos) - 1):
            self.assertLess(y_pos[j], y_pos[j+1])
            self.assertLess(y_pos[j+1], 1.0)
            self.assertLess(0.0, y_pos[j])

        x = np.append(x, np.inf)
        
        y_new = smooth_function_exp(x)
        self.assertEqual(y_new[-1], 1.0)

    def test_matrix(self):

        x = 2 * np.random.random(size=(5, 5)) - 1
        y = smooth_function_exp(x)

        self.assertEqual(y.shape[0], x.shape[0])
        self.assertEqual(y.shape[1], x.shape[1])

        x_flat = x.ravel()
        indices = np.argsort(x_flat)

        xf = x_flat[indices]
        yf = y.ravel()[indices]

        for j in range(len(yf) - 1):
            if xf[j] <= 0:
                self.assertEqual(yf[j], 0.0)
            else:
                self.assertLess(yf[j], yf[j+1])

class TestMollifier(unittest.TestCase):

    def test_trends(self):

        x = np.linspace(-2, 2, 20)
        y = mollifier(x)

        outside_indices = np.abs(x) > 1
        inc_indices = np.logical_and(x >= -1, x <= 0)
        dec_indices = np.logical_and(x > 0, x <= 1)

        self.assertAlmostEqual(np.linalg.norm(y[outside_indices]), 0.0, delta=1e-14)

        y_inc = y[inc_indices]
        y_dec = y[dec_indices]

        for j in range(len(y_dec) - 1):
            self.assertGreater(y_dec[j], y_dec[j+1])

        for j in range(len(y_inc) - 1):
            self.assertLess(y_inc[j], y_inc[j+1])

        self.assertAlmostEqual(mollifier(0), np.exp(-2) ** 2, delta=1e-14)

    def test_unit_interval(self):
        x = np.linspace(-1, 3, 100)
        y = mollifier(x, loc=0.5, scale=0.5)

        outside_indices = np.logical_or(x <= 0, x >= 1)
        inc_indices = np.logical_and(x > 0, x <= 0.5)
        dec_indices = np.logical_and(x > 0.5, x < 1)

        self.assertAlmostEqual(np.linalg.norm(y[outside_indices]), 0.0, delta=1e-14)

        y_inc = y[inc_indices]
        y_dec = y[dec_indices]

        for j in range(len(y_inc) - 1):
            self.assertGreater(y_inc[j+1], y_inc[j])
        for j in range(len(y_dec) - 1):
            self.assertLess(y_dec[j+1], y_dec[j])

class TestIntegralError(unittest.TestCase):

    def test(self):
        from mollifiers.mollifiers import l1_norm_default_mollifier_eval
        area, error = l1_norm_default_mollifier_eval

        self.assertLess(error, 1e-8)
        self.assertLess(error / area, 1e-7)

class TestStep(unittest.TestCase):

    def test_trends(self):
        x = np.linspace(-1, 2, 20)
        zero_indices = x <= 0
        one_indices = x >= 1
        inc_indices = np.logical_and(x > 0, x < 1)

        y = smooth_step(x)

        self.assertAlmostEqual(np.linalg.norm(y[zero_indices]), 0.0, delta=1e-14)
        self.assertAlmostEqual(np.linalg.norm(y[one_indices] - 1), 0.0, delta=1e-14)

        y_inc = y[inc_indices]
        for j in range(len(y_inc) - 1):
            self.assertLess(y_inc[j], y_inc[j+1])
            self.assertLess(y_inc[j+1], 1.0)
            self.assertGreater(y_inc[j], 0.0)

    def test_nonstandard_interval(self):
        new_ending = -1
        new_ident = 2

        x = np.linspace(new_ending - 1, new_ident + 1, 20)
        y = smooth_step(x, end_support=new_ending, begin_identity=new_ident)

        xi = (x - new_ending) / (new_ident - new_ending)
        upsilon = smooth_step(xi)

        self.assertAlmostEqual(np.linalg.norm(upsilon - y), 0.0, delta=1e-14)

    def test_flipped_interval(self):

        new_ending = -1

        x = np.linspace(-2, 2, 20)
        y1 = smooth_step(x, end_support=new_ending)
        y2 = smooth_step(x, end_support=1, begin_identity=new_ending)

        self.assertAlmostEqual(np.linalg.norm(y1 - y2[::-1]), 0.0, delta=1e-14)


class TestIndicator(unittest.TestCase):

    def test_standard(self):
        
        x = np.linspace(-3, 3, 101)
        y = smooth_indicator(x)


        outside_indices = np.logical_or(x <= -2, x >= 2)
        unity_indices = np.abs(x) <= 1
        inc_indices = np.logical_and(x > -2, x < -1)
        dec_indices = np.logical_and(x > 1, x < 2)

        self.assertAlmostEqual(np.linalg.norm(y[outside_indices]), 0.0, delta=1e-14)
        self.assertAlmostEqual(np.linalg.norm(y[unity_indices] - 1), 0.0, delta=1e-14)

        y_inc, y_dec = y[inc_indices], y[dec_indices]

        for j in range(len(y_inc) - 1):
            self.assertGreater(y_inc[j+1], y_inc[j])
            self.assertGreater(y_inc[j], 0)
            self.assertLess(y_inc[j+1], 1)

        for j in range(len(y_dec) - 1):
            self.assertLess(y_dec[j+1], y_dec[j])
            self.assertGreater(y_dec[j+1], 0)
            self.assertLess(y_dec[j+1], 1)

        
        mid_index = len(x) // 2
        self.assertEqual(x[mid_index], 0.0)
        self.assertAlmostEqual(np.linalg.norm(y - y[::-1]), 0.0, delta=1e-14)

    def test_nonstandard_interval(self):

        new_interval = (-2, 5)
        rev_size = 0.1
        x = np.linspace(-3, 6, 1001)
        y = smooth_indicator(x, unity_span=new_interval, rev_up=rev_size, rev_down=rev_size)

        unity_indices = np.logical_and(x >= new_interval[0], x <= new_interval[1])
        self.assertAlmostEqual(np.linalg.norm(y[unity_indices] - 1), 0.0, delta=1e-14)

        zero_indices = np.logical_or(x <= new_interval[0] - rev_size, x >= new_interval[1] + rev_size)
        self.assertAlmostEqual(np.linalg.norm(y[zero_indices]), 0.0, delta=1e-14)

        inc_indices = np.logical_and(x > new_interval[0] - rev_size, x < new_interval[0])
        dec_indices = np.logical_and(x > new_interval[1], x < new_interval[1] + rev_size)
        y_inc = y[inc_indices]
        y_dec = y[dec_indices]

        for j in range(len(y_inc) - 1):
            self.assertGreater(y_inc[j], 0)
            self.assertGreater(y_inc[j+1], y_inc[j])
            self.assertLess(y_inc[j+1], 1)

        for j in range(len(y_dec) - 1):
            self.assertGreater(y_dec[j], 0)
            self.assertLess(y_dec[j+1], y_dec[j])
            self.assertLess(y_dec[j], 1) 

        self.assertEqual(len(y_inc), len(y_dec))
        difference = y_inc - y_dec[::-1]
        self.assertAlmostEqual(np.linalg.norm(difference), 0, delta=1e-13)

if __name__ == "__main__":

    try:
        import matplotlib.pyplot as plt 

        x = np.linspace(-2, 4, 1000)
        y_exp = smooth_function_exp(x)

        x_small = np.linspace(-2, 2, 1000)
        y_mol = mollifier(x_small)

        y_step = smooth_step(x)

        rev = 0.5
        y_1 = smooth_indicator(x_small, rev_down=rev, rev_up=rev)
        
        fig, axes = plt.subplots(nrows=2, ncols=2)
        axes = axes.ravel()

        axes[0].plot(x, y_exp)
        axes[0].set_title("$\infty$'ly differentiable fct")

        axes[1].plot(x_small, y_mol)
        axes[1].set_title("Standard mollifier on [-1, 1]")

        axes[2].plot(x, y_step)
        axes[2].set_title("Smooth step from (0, 0) to (1, 1)")

        axes[3].plot(x_small, y_1)
        axes[3].set_title("Smooth indicator on [-1, 1], with tol 0.5")

        for j in range(len(axes)):
            axes[j].set_xlabel("x")
            axes[j].set_ylabel("y")

        fig.tight_layout()

        plt.savefig("mollifiers_and_friends.pdf")
        plt.show()

    except:
        print("Cannot run demo without matplotlib")