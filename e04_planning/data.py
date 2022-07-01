import mdp as util

__author__ = 'Henning'


def create_mdp_circle_world_one():
    """
    Returns a MarkovDecisionProcess object for Circle-World-One
    :return: A MarkovDecisionProcess
    :rtype: util.MarkovDecisionProcess
    """
    psas = [[[0.00, 0.75, 0.00, 0.00, 0.00, 0.00, 0.00, 0.25],
             [0.00, 0.25, 0.00, 0.00, 0.00, 0.00, 0.00, 0.75]],

            [[0.00, 0.00, 0.75, 0.00, 0.00, 0.00, 0.00, 0.00],
             [0.00, 0.00, 0.25, 0.00, 0.00, 0.00, 0.00, 0.00]],

            [[0.00, 0.25, 0.00, 0.75, 0.00, 0.00, 0.00, 0.00],
             [0.00, 0.75, 0.00, 0.25, 0.00, 0.00, 0.00, 0.00]],

            [[1.00, 0.00, 0.25, 0.00, 0.75, 0.00, 0.00, 0.00],
             [1.00, 0.00, 0.75, 0.00, 0.25, 0.00, 0.00, 0.00]],

            [[0.00, 0.00, 0.00, 0.25, 0.00, 0.75, 0.00, 0.00],
             [0.00, 0.00, 0.00, 0.75, 0.00, 0.25, 0.00, 0.00]],

            [[0.00, 0.00, 0.00, 0.00, 0.25, 0.00, 0.75, 0.00],
             [0.00, 0.00, 0.00, 0.00, 0.75, 0.00, 0.25, 0.00]],

            [[0.00, 0.00, 0.00, 0.00, 0.00, 0.25, 0.00, 0.75],
             [0.00, 0.00, 0.00, 0.00, 0.00, 0.75, 0.00, 0.25]],

            [[0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.25, 0.00],
             [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.75, 0.00]]]
    ras = [[4096., -512., 0., 0., 0., 0., 0., 0.],
           [4096., -512., 0., 0., 0., 0., 0., 0.]]
    state_start = 4
    gamma = 0.5
    return util.MarkovDecisionProcess(ras=ras, psas=psas, state_start=state_start, gamma=gamma)


if __name__ == '__main__':  # Running data.py will validates and prints all data
    mdp = create_mdp_circle_world_one()
    print '================================='
    print 'MDP Circle World One:'
    print '================================='
    print mdp.psas
    print mdp.ras
