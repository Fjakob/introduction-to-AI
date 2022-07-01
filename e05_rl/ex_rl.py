import matplotlib.pyplot as plt
import numpy as np
from environment import environment

def eps_greedy_action(Q, s, eps):
    '''
    Implement epsilon greedy action selection:
        With Probability 1-eps select greedy action,
        otherwise select random action

    :param Q: Q-function
    :type Q: numpy.ndarray
    :param s: state 
    :type s: int 
    :param eps: epsilon for eps-greedy action selection 
    :type eps: float 
    
    :returns: selected action
    :rtype: int
    '''
    if np.random.rand() > eps:
        return np.argmax(Q[s, :]) 
    else:
        return np.random.randint(0, Q.shape[1]) 


def smooth(y, box_pts):
    '''
    Simple box filter implementation for smoothing
    
    :param y: discrete signal to be smoothed
    :type gamma: npumpy.ndarray
    
    :returns: smoothed signal 
    :rtype: numpy.ndarray
    '''

    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth

def qlearning(Q, s0, env, alpha, gamma, eps, episodes):
    '''
    Implementation of the Q-Learning algorithm as explained in the slides
    
    :param Q: Q-function
    :type Q: numpy.ndarray
    :param s0: initial state 
    :type s0: int 
    :param env: environt 
    :type env: function 
    :param alpha: learning rate 
    :type alpha: float 
    :param gamma: discount factor 
    :type gamma: float 
    :param eps: epsilon for eps-greedy action selection 
    :type eps: float 
    :param episodes: number of episodes 
    :type eps: int

    :returns: Q, list of the commulative reward per episode 
    :rtype: tuple
    '''
    rewardList = np.array([])
    for t in range(episodes):
        s = s0
        abs_reward = 0
        while(True): #do until goal is reached
            a = eps_greedy_action(Q, s, eps)
            s_new, reward = env.apply_action(s, a)
            abs_reward += reward
            Q[s, a] += alpha*(reward + gamma * max(Q[s_new]) - Q[s, a])
            s = s_new
            if s is env.goal:
                break  
        rewardList = np.append(rewardList, abs_reward) #end of episode
        
    return Q, rewardList

def sarsa(Q, s0, env, alpha, gamma, eps, episodes):
    '''
    Implementation of the Q-Learning algorithm as explained in the slides
    
    :param Q: Q-function
    :type Q: numpy.ndarray
    :param s0: initial state 
    :type s0: int 
    :param env: environt 
    :type env: function 
    :param alpha: learning rate 
    :type alpha: float 
    :param gamma: discount factor 
    :type gamma: float 
    :param eps: epsilon for eps-greedy action selection 
    :type eps: float 
    :param episodes: number of episodes 
    :type episodes: int

    :returns: Q, list of the commulative reward per episode 
    :rtype: tuple
    '''
    rewardList = np.array([])
    for t in range(episodes):
        s = s0
        abs_reward = 0
        a = eps_greedy_action(Q, s, eps)
        while(True):
            s_new, reward = env.apply_action(s, a)
            abs_reward += reward
            a_ = np.argmax(Q[s_new, :])
            Q[s, a] += alpha*(reward + gamma * Q[s_new, a_] - Q[s, a])
            s = s_new
            a = a_
            if s is env.goal:
                break  
            #do until goal is reached
        rewardList = np.append(rewardList, abs_reward)
            
    return Q, rewardList

def qlearning_sched(Q, s0, env, alpha, gamma, eps0, episodes):
    '''
    Implementation of the Q-Learning algorithm as explained in the slides
    
    :param Q: Q-function
    :type Q: numpy.ndarray
    :param s0: initial state 
    :type s0: int 
    :param env: environt 
    :type env: function 
    :param alpha: learning rate 
    :type alpha: float 
    :param gamma: discount factor 
    :type gamma: float 
    :param eps: epsilon for eps-greedy action selection 
    :type eps: float 
    :param episodes: number of episodes 
    :type eps: int

    :returns: Q, list of the commulative reward per episode 
    :rtype: tuple
    '''
    rewardList = np.array([])
    for t in range(episodes):
        s = s0
        abs_reward = 0
        while(True):
            a = eps_greedy_action(Q, s, eps0)
            s_new, reward = env.apply_action(s, a)
            abs_reward += reward
            Q[s, a] += alpha*(reward + gamma * max(Q[s_new]) - Q[s, a])
            s = s_new
            if s is env.goal:
                break  
            #do until goal is reached
        rewardList = np.append(rewardList, abs_reward)
        eps0 *= 0.5
        
        
    return Q, rewardList

def sarsa_sched(Q, s0, env, alpha, gamma, eps0, episodes):
    '''
    Implementation of the Q-Learning algorithm as explained in the slides
    
    :param Q: Q-function
    :type Q: numpy.ndarray
    :param s0: initial state 
    :type s0: int 
    :param env: environt 
    :type env: function 
    :param alpha: learning rate 
    :type alpha: float 
    :param gamma: discount factor 
    :type gamma: float 
    :param eps: epsilon for eps-greedy action selection 
    :type eps: float 
    :param episodes: number of episodes 
    :type episodes: int

    :returns: Q, list of the commulative reward per episode 
    :rtype: tuple
    '''
    rewardList = np.array([])
    for t in range(episodes):
        s = s0
        abs_reward = 0
        a = eps_greedy_action(Q, s, eps0)
        while(True):
            s_new, reward = env.apply_action(s, a)
            abs_reward += reward
            a_ = eps_greedy_action(Q, s, eps0)
            Q[s, a] += alpha*(reward + gamma * Q[s_new, a_] - Q[s, a])
            s = s_new
            a = a_
            if s is env.goal:
                break  
            #do until goal is reached
        rewardList = np.append(rewardList, abs_reward)
        eps0 *= 0.5
        
        
    return Q, rewardList


if __name__ == "__main__":
    env = environment()
    
    #Params
    episodes = 5000
    alpha = .1
    gamma = 1.
    eps = .1
    eps0 = 1.
    s0 = 36

    Q_ql, R_ql = qlearning(np.zeros((48, 4)), s0, env, alpha, gamma, eps, episodes)
    Q_sa, R_sa = sarsa(np.zeros((48, 4)), s0, env , alpha, gamma, eps, episodes)
    #Q_ql_sched, R_ql_sched = qlearning_sched(np.zeros((48, 4)), s0, env, alpha, gamma, eps0, episodes)
    #Q_sa_sched, R_sa_sched = sarsa_sched(np.zeros((48, 4)), s0, env , alpha, gamma, eps0, episodes)

    print("Q_Learning")
    env.print_greedy_policy(Q_ql)
    print("Sarsa")
    env.print_greedy_policy(Q_sa)
    #print("Q_Learning with scheduled eps")
    #env.print_greedy_policy(Q_ql_sched)
    #print("Sarsa with scheduled eps")
    #env.print_greedy_policy(Q_sa_sched)

    #np.savetxt("R_ql.csv", R_ql)
    #np.savetxt("R_sa.csv", R_sa)
    #np.savetxt("R_ql_sched.csv", R_ql_sched)
    #np.savetxt("R_sa_sched.csv", R_sa_sched)

    plt.plot(smooth(R_ql, 50), label='Q-Learning')
    plt.plot(smooth(R_sa, 50), label='Sarsa')
    #plt.plot(smooth(R_ql_sched, 50), label='Q-Learning with scheduled eps')
    #plt.plot(smooth(R_sa_sched, 50), label='Sarsa with scheduled eps')
    plt.legend(loc=4)
    plt.show()
