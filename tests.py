import unittest
import kowalsky as kow
import matplotlib.pyplot as plt
import numpy as np
import numbers


class to_np(unittest.TestCase):
    def test_nparray(self):
        self.assertIsInstance(kow.to_np(np.array([2, 3])), np.ndarray)

    def test_arr(self):
        self.assertIsInstance(kow.to_np([2, 3]), np.ndarray)

    def test_list(self):
        self.assertIsInstance(kow.to_np((2, 3, 4)), np.ndarray)

    def test_num(self):
        self.assertIsInstance(kow.to_np(5.7), numbers.Number)

    def test_fail(self):
        with self.assertRaises(ValueError):
            to_np("hi")


class update_dict(unittest.TestCase):
    def test_error(self):
        with self.assertRaises(ValueError):
            kow.update_dict({'a': 1, 'b': 2}, c=3)

    def test_update(self):
        self.assertAlmostEqual(kow.update_dict({'a': 1, 'b': 2}, a=3),
                               {'a': 3, 'b': 2})


class dist(unittest.TestCase):
    def test_init(self):
        with self.assertRaises(TypeError):
            kow.dist('bla bla bla')

    def test_generic(self):
        def f(x, y):
            return x + y
        d = kow.dist('generic', func=f, y=4)
        self.assertEqual(d.type, 'generic')
        self.assertEqual(d.f(2), 6)

    def test_normalDist(self):
        d = kow.dist('normal')
        self.assertEqual(d.type, 'normal')
        self.assertEqual(d.f(0), kow.normDist(0))
        d = kow.dist('normal', mean=2.3, sd=0.3)
        self.assertEqual(d.f(0), kow.normDist(0, 2.3, 0.3))

    def test_expDist(self):
        d = kow.dist('exp')
        self.assertEqual(d.type, 'exp')
        self.assertEqual(d.f(0), kow.expDist(0))
        d = kow.dist('exp', mean=2.3)
        self.assertEqual(d.f(0), kow.expDist(0, 2.3))


def test_plots():
    fig, ax = plt.subplots()
    d = kow.dist('exp')
    print(d.par)
    d.plot(ax, 0, 4, 100)
    plt.show()


if __name__ == '__main__':
    unittest.main()
    test_plots()
