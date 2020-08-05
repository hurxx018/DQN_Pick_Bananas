

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class DQNAgent():

    def __init__(self):
        super(DQNAgent, self).__init__()


    def step(
        self,
        state,
        action,
        reward,
        next_state, 
        done
        ):
        pass

    def act(
        self,
        state,
        epsilon = 0. 
        ):
        pass

    def learn(
        self
        ):
        pass

    def soft_update(
        self
        ):
        pass