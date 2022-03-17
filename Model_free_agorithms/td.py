#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#GokuL Srinivasan
import numpy as np
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Temporal Difference
    In this problem, you will implememnt an AI player for cliffwalking.
    The main goal of this problem is to get familar with temporal diference algorithm.
    You could test the correctness of your code 
    by typing 'nosetests -v td_test.py' in the terminal.
    
    You don't have to follow the comments to write your code. They are provided
    as hints in case you need. 
'''
#-------------------------------------------------------------------------

def epsilon_greedy(Q, state, nA, epsilon = 0.1):
    """Selects epsilon-greedy action for supplied state.
    
    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    state: int
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1
    
    Returns:
    --------
    action: int
        action based current state
     Hints:
        You can use the function from project2-1
    """
    ############################
    
    # YOUR IMPLEMENTATION HERE #
    action = 0
    if np.random.uniform() > epsilon:
        action = np.argmax(Q[state])
    else:
        action = np.random.choice(nA)

    ############################
    return action

def sarsa(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """On-policy TD control. Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    Hints:
    -----
    You could consider decaying epsilon, i.e. epsilon = 0.99*epsilon during each episode.
    """
    
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    ############################# 

    for episode in range(n_episodes):

        epsilon = 0.99 * epsilon

        state = env.reset()
        nA = env.action_space.n
        action = epsilon_greedy(Q, state, nA, epsilon)

        while (True):
        
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, nA, epsilon)
            Q[state][action] += alpha * (reward + gamma * (Q[next_state][next_action] - Q[state][action]))
            action = next_action
            if done: 
                break

            action = next_action
            state = next_state
                
    ############################
    return Q

def q_learning(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """Off-policy TD control. Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    """
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    ############################
    # YOUR IMPLEMENTATION HERE #
    epsilon_start = 0.1
    epsilon_end = 0.01

    for episode in range(n_episodes):
        epsilon = epsilon_start - (episode * (epsilon_start - epsilon_end)) / n_episodes
        state = env.reset()
        nA = env.action_space.n
        
        while (True):
            action = epsilon_greedy(Q, state, nA, epsilon)

            next_state, reward, done, _ = env.step(action)
            Q[state][action] += alpha * (reward + gamma * (max(Q[next_state]) - Q[state][action]))
            
            if done: 
                break

            state = next_state 

    ############################
    return Q