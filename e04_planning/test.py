import ex_qlearn as ex
import data as data
import unittest
import numpy as np

__author__ = 'Henning'


class VIterationTest(unittest.TestCase):
    def setUp(self):
        self.mdp = data.create_mdp_circle_world_one()
        ex.PRINTING = False

    def test_return_type(self):
        v, q = ex.value_iteration(self.mdp, 1)
        self.assertIsNotNone(v)
        self.assertIsNotNone(q)
        self.assertTrue(type(v) is np.ndarray)
        self.assertTrue(type(q) is np.ndarray)
        self.assertTupleEqual(v.shape, (8L,))
        self.assertTupleEqual(q.shape, (8L, 2L))

    def test_first_steps(self):
        q = [None, None, None, None]
        _, q[0] = ex.value_iteration(self.mdp, 1)
        _, q[1] = ex.value_iteration(self.mdp, 2)
        _, q[2] = ex.value_iteration(self.mdp, 3)
        _, q[3] = ex.value_iteration(self.mdp, 4)
        self.assertIsNotNone(q[0])
        self.assertIsNotNone(q[1])
        self.assertIsNotNone(q[2])
        self.assertIsNotNone(q[3])
        q = np.array(q)
        self.assertTupleEqual(q.shape, (4L, 8L, 2L))
        h = (np.array(q).transpose(0, 2, 1)
             .dot(np.array([2, 3, 5, 7, 11, 13, 17, 19]))
             .dot(np.array([23, 29]))
             .dot(np.array([31, 37, 41, 43])))
        if h != 246416958.0:
            self.fail("VIterationTest: In at least one of the first 4 step the q-function is wrong")

    def test_convergence(self):
        v_is, q_is = ex.value_iteration(self.mdp, 100)
        atol = 1e-4
        rtol = 1e-6
        q_set = np.array([[4185.93497537, 4185.93497537],
                          [1112.69359606, 176.14581281],
                          [439.74384236, 206.53793103],
                          [179.86995074, 99.86600985],
                          [99.86600985, 119.72807882],
                          [126.34876847, 259.3182266],
                          [303.64137931, 651.60591133],
                          [767.59408867, 1651.17635468]])
        v_set = np.array([4185.93497537, 1112.69359606, 439.74384236, 179.86995074,
                          119.72807882, 259.3182266, 651.60591133, 1651.17635468])
        self.assertIsNotNone(q_is)
        self.assertTrue(type(q_is) is np.ndarray)
        self.assertTupleEqual(q_set.shape, q_is.shape)
        self.assertIsNotNone(v_is)
        self.assertTrue(type(v_is) is np.ndarray)
        self.assertTupleEqual(v_is.shape, v_set.shape)
        q_dif = q_set - q_is
        for dif in q_dif.flat:
            if -atol < dif < -atol:
                q_dif[...] = 0
        q_dif = np.divide(q_dif, q_set)
        for dif in q_dif.flat:
            self.assertTrue(-rtol < dif < rtol, "QLearningTest: q-function is outside the tolerance")
        v_dif = v_set - v_is
        for dif in v_dif.flat:
            if -atol < dif < -atol:
                v_dif[...] = 0
        v_dif = np.divide(v_dif, v_set)
        for dif in v_dif.flat:
            self.assertTrue(-rtol < dif < rtol, "QLearningTest: v-function is outside the tolerance")


class QLearningTest(unittest.TestCase):
    def setUp(self):
        self.mdp = data.create_mdp_circle_world_one()
        ex.util.random_seed(0)
        ex.PRINTING = False

    def test_return_type(self):
        v, q = ex.qlearning(self.mdp, alpha=0.01, steps=1)
        self.assertIsNotNone(v)
        self.assertIsNotNone(q)
        self.assertTrue(type(v) is np.ndarray)
        self.assertTrue(type(q) is np.ndarray)
        self.assertTupleEqual(v.shape, (8L,))
        self.assertTupleEqual(q.shape, (8L, 2L))

    def test_convergence(self):
        v_is, q_is = ex.qlearning(self.mdp, alpha=0.01, steps=100000)
        atol = 1e+1
        rtol = 5e-1
        q_set = np.array([[4189.15053499, 4189.08195324],
                          [1174.75345199, 209.88473187],
                          [485.0963055, 209.7549114],
                          [195.2073548, 95.45178875],
                          [101.77227013, 113.84838772],
                          [130.7373347, 243.21867168],
                          [279.29343183, 630.20484408],
                          [676.02962682, 1579.7475817]])
        v_set = np.array([4189.15053499, 1174.75345199, 485.0963055, 195.2073548,
                          113.84838772, 243.21867168, 630.20484408, 1579.7475817])
        self.assertIsNotNone(q_is)
        self.assertTrue(type(q_is) is np.ndarray)
        self.assertTupleEqual(q_set.shape, q_is.shape)
        self.assertIsNotNone(v_is)
        self.assertTrue(type(v_is) is np.ndarray)
        self.assertTupleEqual(v_is.shape, v_set.shape)
        q_dif = q_set - q_is
        for dif in q_dif.flat:
            if -atol < dif < -atol:
                q_dif[...] = 0
        q_dif = np.divide(q_dif, q_set)
        for dif in q_dif.flat:
            self.assertTrue(-rtol < dif < rtol, "QLearningTest: q-function is outside the tolerance")
        v_dif = v_set - v_is
        for dif in v_dif.flat:
            if -atol < dif < -atol:
                v_dif[...] = 0
        v_dif = np.divide(v_dif, v_set)
        for dif in v_dif.flat:
            self.assertTrue(-rtol < dif < rtol, "QLearningTest: v-function is outside the tolerance")
