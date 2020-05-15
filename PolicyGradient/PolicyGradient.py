#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, losses, optimizers
from tensorflow.keras import backend as K


# In[2]:


class PolicyGradient:
    def __init__(self, observationCount, actionCount, rewardDecay=0.95, learningRate = 0.01):
        self.actionCount = actionCount
        self.rewardDecay = rewardDecay
        self.episodeObservations, self.episodeActions, self.episodeRewards = [], [], []
        
        inputs = layers.Input(shape=(observationCount,))
        layer01 = layers.Dense(10, activation='tanh')(inputs)
        outputs = layers.Dense(actionCount, activation='tanh')(layer01)
        outputs = keras.activations.softmax(outputs) 
        
        actionsValue = layers.Input(shape=(None,))
        
        optimizer = keras.optimizers.Adam(learning_rate=learningRate)
        
        self.nn = models.Model(inputs=[inputs, actionsValue], outputs=outputs)
        self.nn.compile(loss=self.Loss(actionsValue), optimizer=optimizer, experimental_run_tf_function=False)
        
    def Loss(self, actionsValue):
        def Func(y_true, y_pred):
            v = keras.losses.sparse_categorical_crossentropy(y_true, y_pred, True)
            av = K.squeeze(actionsValue, axis=-1)
            v = tf.reduce_mean(v * av)
            return v
        return Func
    
    def StoreTransition(self, observation, action, reward):
        self.episodeObservations.append(observation)
        self.episodeActions.append(action)
        self.episodeRewards.append(reward)
        
    def Learn(self):
        actionsValue = np.array(self.DiscountAndNormRewards())
        tmpObservations = np.array(self.episodeObservations)
        tmpActions = np.array(self.episodeActions)  
        self.nn.fit((tmpObservations, actionsValue), tmpActions)        
        self.episodeObservations, self.episodeActions, self.episodeRewards = [], [], []
        return actionsValue
    
    def ChooseAction(self, observation):
        observation = np.expand_dims(observation, 0)
        weights =  self.nn.predict((observation, np.array([])))
        action = np.random.choice(range(weights.shape[1]), p=weights.ravel())
        return action
        
    def DiscountAndNormRewards(self):
        discounted_ep_rs = np.zeros_like(self.episodeRewards)
        running_add = 0
        for t in reversed(range(0, len(self.episodeRewards))):
            running_add = running_add * self.rewardDecay + self.episodeRewards[t]
            discounted_ep_rs[t] = running_add
        n = (discounted_ep_rs - np.mean(discounted_ep_rs)) / np.std(discounted_ep_rs)
        return n
    
    def Show(self):
        self.nn.summary()


# In[3]:


import gym
import matplotlib.pyplot as plt


# In[4]:


DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

env = gym.make('CartPole-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

RL = PolicyGradient(
    actionCount=env.action_space.n,
    observationCount=env.observation_space.shape[0]
)

for i_episode in range(30000):

    observation = env.reset()

    while True:
        if RENDER: env.render()

        action = RL.ChooseAction(observation)

        observation_, reward, done, info = env.step(action)

        RL.StoreTransition(observation, action, reward)

        if done:
            ep_rs_sum = sum(RL.episodeRewards)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))

            vt = RL.Learn()

            if i_episode == 0:
                plt.plot(vt)    # plot the episode vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            break

        observation = observation_


# In[ ]:




