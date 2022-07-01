import numpy as np
from random import Random
import data as data

__author__ = 'Henning'


class InvaildArgumentException(Exception):
    pass


class MarkovDecisionProcess(object):
    """
    The main MDP data structure.

    Examples:
    mdp.ps[1]  # returns the start distribution for state one
    mdp.psas[1,0,2]  # return the transition from state 2 to state 1 given action 0
    mdp.ras[0,1]  # return the reward for doing action 0 in state 1 (the state it was before the action)

    :param num_actions: number of states
    :param num_states: number of actions
    :param state_start: start state
    :param psas: transition probabilities
    :param ras: reward expectation as a function of (action,x_before)
    :param gamma: discounting factor
    :type num_actions: int
    :type num_states: int
    :type state_start: int
    :type psas: np.ndarray
    :type ras: np.ndarray
    :type gamma: float
    """

    def __init__(self, ras, psas, state_start, gamma):
        """
        The constructor for a MDP.
        :param state_start: start distribution as nested list
        :param psas: transition probabilities as nested list
        :param ras: reward expectation as a function of (action,x_before) as nested list
        :param gamma: discounting factor

        :type state_start: int
        :type psas: list
        :type ras: list
        :type gamma: float
        """

        self.gamma = gamma
        self.ras = self.__check_input_tensor(
            ras,
            req_shape=(None, None)
        )
        self.num_actions = self.ras.shape[0]
        self.num_states = self.ras.shape[1]
        self.psas = self.__check_input_tensor(
            psas,
            normalized=True,
            req_shape=(self.num_states, self.num_actions, self.num_states)
        )
        self.state_start = state_start

    @staticmethod
    def __check_input_tensor(input_tensor, req_shape=None, normalized=False):
        """
        Does some check on the input for the MDP
        :param input_tensor: the input to check as nested list
        :param req_shape: the required shape as tuple or None for anything.
                          The tuple may have None or any integer as entry.
        :param normalized: whether the input should be is a distribution over the first index w.r.t. the later indices
        :return: the input as numpy.ndarray
        :type input_tensor: list
        :type req_shape: tuple
        :type normalized: bool
        :rtype: np.ndarray
        """
        arr = input_tensor
        if type(input_tensor) is not np.ndarray:
            if type(input_tensor) is not list:
                raise InvaildArgumentException  # input must be a list
            arr = np.array(input_tensor, np.float64)

        if req_shape is not None:  # check the shape
            if arr.ndim != len(req_shape):
                raise InvaildArgumentException  # input does not have the right dimensions
            for i in range(len(req_shape)):
                if (req_shape[i] is not None) & (arr.shape[i] != req_shape[i]):
                    raise InvaildArgumentException  # input does not have the required shape

        if normalized:  # check whether the input is a distribution over the first index w.r.t. the later indices
            if arr.ndim == 1:
                if not (1 - sum(arr)) < 1e-10:
                    raise InvaildArgumentException  # input is not a distribution over the first index
            elif arr.ndim == 2:
                for j in range(arr.shape[0]):
                    if not (1 - sum(arr[:, j])) < 1e-10:
                        raise InvaildArgumentException  # input is not a distribution over the first index
            elif arr.ndim == 3:
                for k in range(arr.shape[2]):
                    for j in range(arr.shape[1]):
                        if not (1 - sum(arr[:, j, k])) < 1e-10:
                            raise InvaildArgumentException  # input is not a distribution over the first index
            elif arr.ndim == 4:
                for l in range(arr.shape[3]):
                    for k in range(arr.shape[2]):
                        for j in range(arr.shape[1]):
                            if not (1 - sum(arr[:, j, k, l])) < 1e-10:
                                raise InvaildArgumentException  # input is not a distribution over the first index
            else:
                raise Exception  # Not implemented
        return arr

__rnd = Random()

random_seed = __rnd.seed


def sample_multinomial(p):
    """
    Gets a sample form given multinomial distribution

    The distribution is an 1-D-array containing the probability as entries.
    The sample will be given as the index.

    :param p: the distribution
    :return: the sample
    :type p: np.array
    :rtype: int
    """
    if (type(p) is not list) and (type(p) is not tuple):
        if type(p) is not np.ndarray:
            raise InvaildArgumentException  # p must be a numpy.ndarray or a list or a tuple
        if p.ndim != 1:
            raise InvaildArgumentException  # p must have just 1 dimension
    if (1 - sum(p)) >= 1e-10:
        raise InvaildArgumentException  # p must be normalized
    s = 0.0
    ptr = __rnd.uniform(0.0, 1.0)
    for i in range(len(p)):
        s += p[i]
        if s > ptr:
            return i
    raise Exception  # Unexpected error, maybe p is not normalized



def random_range(start, stop=None, step=1):
    """
    Get a random integer from range(start, stop, step)

    :param start: lower bound
    :param stop: upper bound
    :param step: step
    :return: random integer
    :type start: int
    :type stop: int
    :type step: int
    :rtype: int
    """
    return __rnd.randrange(start, stop, step)


def random_uniform(a=0.0, b=1.0):
    """
    Get a random number in the range [a, b) or [a, b] depending on rounding.

    :param a: lower bound
    :param b: upper bound
    :return: random number
    :type a: float
    :type b: float
    :rtype: float
    """
    return __rnd.uniform(a, b)


def random_gaussian(mu=0.0, sigma=1.0):
    """
    Gaussian distribution

    :param mu: mean
    :param sigma: standard deviation
    :return: a random gaussian
    :type mu: float
    :type sigma: float
    :rtype: float
    """
    return __rnd.gauss(mu, sigma)


def random_tensor_uniform(shape, a=0.0, b=1.0):
    """
    Get a random tensor using a uniform distribution

    :param shape: the shape of the tensor
    :param a: lower bound
    :param b: upper bound
    :return: the random tensor
    :type shape: tuple:
    :type a: float
    :type b: float
    :rtype: np.ndarray
    """
    r = np.empty(shape, np.float64)
    for _ in r.flat:
        r[...] = __rnd.uniform(a, b)
    return r


def random_tensor_gaussian(shape, mu=0.0, sigma=1.0):
    """
    Get a random tensor using a Gaussian distribution

    :param shape: the shape of the tensor
    :param mu: mean
    :param sigma: standard deviation
    :return: the random tensor
    :type shape: tuple
    :type mu: float
    :type sigma: float
    :rtype: np.ndarray
    """
    r = np.empty(shape, np.float64)
    for _ in r.flat:
        r[...] = __rnd.gauss(mu, sigma)
    return r


def argmax(p):
    """
    Gets the index of maximum of p

    :param p: the array
    :return: the index of the maximum
    :type p: np.array
    :rtype: int
    """
    if (type(p) is not list) and (type(p) is not tuple):
        if type(p) is not np.ndarray:
            raise InvaildArgumentException  # p must be a numpy.ndarray or a list or a tuple
        if p.ndim != 1:
            raise InvaildArgumentException  # p must have just 1 dimension

    m = 0
    for i in range(len(p)):
        if p[i] > p[m]:
            m = i
    return m
