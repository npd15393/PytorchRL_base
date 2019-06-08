import torch 
import gym
import copy
import random as rand
import numpy as np
from torch.distributions import *

class ppo_net(torch.nn.Module):
    def __init__(self,n_f,n_a):
        super(ppo_net,self).__init__()

    def forward(self):
        pass


class ppo_agent:
    def __init__(self):
        super(PGAgent,self).__init__()
        # self.buff=ReplayMemory(MEM_SIZE)
        self.n_features= env.observation_space.shape[0]
        self.n_actions=env.action_space.n

        self.model=AC_net(self.n_features,self.n_actions)
        # self.epsilon=0.1
        self.lr=1e-5
        self.opt=torch.optim.Adam(self.model.parameters(), lr=self.lr)


    def train(self):
        pass
