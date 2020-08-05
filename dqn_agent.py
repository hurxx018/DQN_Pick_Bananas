

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-3               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNAgent(object):

    def __init__(
        self,
        state_size,
        action_size,
        seed
        ):
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


class ReplayBuffer(object):

    def __init__(
        self,
        action_size,
        buffer_size,
        batch_size, 
        seed
        ):
        pass


    def add(
        self,
        state,
        action, reward,
        next_state, 
        done
        ):
        pass

    def sample(
        self
        ):
        pass

    def __len__(self):
        pass