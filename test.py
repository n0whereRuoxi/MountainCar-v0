import argparse
import os

import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from time import time
import keras as K
envName = "MountainCar-v0"
env = gym.make(envName)

def play_model(actor):
    state = env.reset()
    qs = []
    score = 0
    done = False
    while not done:
        # env.render()
        state = np.reshape(state, [-1, env.observation_space.shape[0]])
        q_actions = actor.predict(state)
        action = np.argmax(q_actions)
        q_max = np.amax(q_actions)
        qs.append(q_max)
        nextState, reward, done, _ = env.step(action)
        state = nextState
        score += reward
        if done:
            return score, qs
    return 0, None

model = "experiments/exp1/save/MountainCar-v0_local_model_1573103728.728163.h5"
totalIters = 100
expectedReward = -110

#Test
testScores = []
mses = []
actor = K.models.load_model('{}'.format(model))
print("Saved model loaded from '{}'".format(model))
print("Starting testing.. Expecting reward to be {} over {} iterations".format(
    expectedReward, totalIters))
for itr in range(1, totalIters + 1):
    score, qs = play_model(actor)
    testScores.append(score)
    qs_ = [-i for i in range(len(qs) - 1, -1, -1)]
    mse = ((np.array(qs) - np.array(qs_))**2).mean()
    mses.append(mse)
    print("Iteration: {}\tScore: {}\tMSE: {}".format(itr, score, mse))

avg_reward = np.mean(testScores)
print("Total Avg. Score over {} consecutive iterations : {}. MSE = {}.".format(totalIters, avg_reward, np.array(mses).mean()))
if avg_reward >= expectedReward:
    print("Agent finished test within expected reward boundary! Environment is solved.")
else:
    print("Agent has failed this test. Average score observed was {}".format(avg_reward))
