import numpy as np
import mdp as util


def print_v_func(k, v):
    if PRINTING:
        print "k={} V={}".format(k, v)


def print_simulation_step(state_old, action, state_new, reward):
    if PRINTING:
        print "s={} a={} s'={} r={}".format(state_old, action, state_new, reward)


def value_iteration(mdp, num_iterations=10):
    """
    Does value iteration on the given Markov Decision Process for given number of iterations

    :param mdp: the Markov Decision Process
    :param num_iterations: the number of iteration to compute the V-function
    :return: (v, q) = the V-function and Q-function after 'num_iterations' steps

    :type mdp: util.MarkovDecisionProcess
    :type num_iterations: int
    """
    possibilities = {} # whack psas structure...
    for state in range(mdp.num_states):
        state_entry = []
        for a in range(mdp.num_actions):
            action_entry = []
            for s in range(mdp.num_states):
                prob = mdp.psas[s][a][state]
                if prob != 0. :
                    action_entry.append([s, prob])
            state_entry.append(action_entry)
        possibilities[state] = state_entry
        
    q = np.zeros((mdp.num_states, mdp.num_actions))  # init q0
    v = np.zeros(mdp.num_states)  # init v0 
    q_old = np.zeros((mdp.num_states, mdp.num_actions))
    
    for k in range(num_iterations):
        print_v_func(k, v)  # print k and v
        
        # Q-function:
        # update q_old with q: (does the same as "q_old = q")
        i = 0
        for s in q:
            j = 0
            for entry in q[i]:           
                q_old[i][j] = entry    # "q_old = q" doesnt work as it should
                j+=1
            i+=1
            
        for s in range(mdp.num_states):
            for a in range(mdp.num_actions):
                reward = mdp.ras[a,s]
                possible_states = possibilities[s][a] # tuple of all s' with probility P(s'|s,a)
                summe = 0
                for next_state, prob in possible_states:  #each summand
                    summe += prob * max(q_old[next_state])
                    if prob == 1:  
                        break #prob = 1 => only one action to reach this state
                q[s][a] = reward + mdp.gamma * summe  #Q-Iteration formula   
        
        # V-function:
        for s in range(mdp.num_states):
            v[s] = max(q[s])   
            
    return v, q


def simulate(mdp, state_old, action):
    """
    Simulates a single step in the given Markov Decision Process

    :param mdp: the Markov Decision Process
    :param action: the Action to be taken
    :param state_old: the old state
    :return: (reward, state_new) = the reward for taking the action in old state and the new state you are in

    :type mdp: util.MarkovDecisionProcess
    :type action: int
    :type state_old: int
    :rtype: tuple
    """
    # this method work as is, no change required
    reward = mdp.ras[action, state_old]  # gets reward
    state_new = util.sample_multinomial(mdp.psas[:, action, state_old])  # get new state as sample s' from P(s'|a,s)
    #print_simulation_step(state_old, action, state_new, reward)  # print transition and reward
    return reward, state_new


def qlearning(mdp, alpha=0.01, steps=1000000):
    """
    Performs Q-Learning on given Markov Decision Process with given for given number of steps

    :param mdp: the Markov Decision Process
    :param alpha: the learning parameter alpha
    :param steps: the number of steps to simulate and learn
    :return:

    :type mdp: util.MarkovDecisionProcess
    :type alpha: float
    :type steps: int
    :rtype:
    """
    # init Q-Function
    q = np.zeros((mdp.num_states, mdp.num_actions))
    state_old = mdp.state_start
    
    for t in range(steps):
        states = []
        for a in range(mdp.num_actions):
            reward, state_new = simulate(mdp, state_old, a)
            q[state_old, a] += alpha*(reward + mdp.gamma * max(q[state_new]) - q[state_old, a])
            states.append(state_new)
            
        state_old = states[0]
        for state in states:
            if max(q[state]) > max(q[state_old]):
                state_old = state
        
    # compute the V-function
    v = np.zeros(mdp.num_states)
    for s in range(mdp.num_states):
        v[s] = max(q[s])

    print_v_func('done', v)  # prints your V-function
    return v, q


PRINTING = False  # do not print by default, please do not change this
if __name__ == '__main__':
    PRINTING = True  # enable printing
    util.random_seed()  # seed random number generator
    mdp = util.data.create_mdp_circle_world_one()
    value_iteration(util.data.create_mdp_circle_world_one())
    qlearning(util.data.create_mdp_circle_world_one())
    