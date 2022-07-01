# -*- coding: utf-8 -*-
import sys
import numpy as np

class environment:
    '''
    Cliffworld environment

    states:
        +--+--+--+--+--+--+--+--+--+--+--+--+
        | 0| 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|
        +--+--+--+--+--+--+--+--+--+--+--+--+
        |12|13|14|15|16|17|18|19|20|21|22|23|
        +--+--+--+--+--+--+--+--+--+--+--+--+
        |24|25|26|27|28|29|30|31|32|33|34|35|
        +--+--+--+--+--+--+--+--+--+--+--+--+
        |36|37|38|39|40|41|42|43|44|45|46|47|
        +--+--+--+--+--+--+--+--+--+--+--+--+
    
    actions:
        a = 0 -> top
        a = 1 -> down
        a = 2 -> left
        a = 3 -> right
    '''

    def __init__(self):
        self.height = 4
        self.width = 12 
        self.cliff = np.arange(37, 47)
        self.start = 36
        self.goal = 47
        self.move_symbols = ['↑', '↓', '←', '→']
        #self.move_symbols = ['u', 'd', 'l', 'r']
        self.r_step = -1
        self.r_cliff = -100

    def apply_action(self, s, a):
        '''
        Assumes current state s and applies action a.
        Returns resulting state and reward.
        '''

        x = s % self.width 
        y = int(s / self.width)
       
        # move
        if a == 0:
            y -= 1 
        elif a == 1:
            y += 1 
        elif a == 2:
            x -= 1
        elif a == 3:
            x += 1

        s_ = s
        r = self.r_step 

        # check if move is legal
        if x < self.width and x >= 0 and y < self.height and y >= 0:
            s_ = y * self.width + x 

            if s_ in self.cliff:
                s_ = self.start
                r = self.r_cliff

        return s_, r

    
    def print_greedy_policy(self, Q):
        '''
        Evaluates policy greedily on Q and prints it.
        
        :param Q: Q-function
        :type Q: numpy.ndarray
        '''
        
        a_seq = np.argmax(Q, axis=1)

        for s in range(a_seq.shape[0]):
            if s in self.cliff:
                sys.stdout.write('#')
            else:
                sys.stdout.write(self.move_symbols[a_seq[s]])

            if (s + 1) % self.width == 0:
                print
        print

env = environment()
s, r = env.apply_action(28,1)
print(s, r)



