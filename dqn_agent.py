

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import QNetworks, weights_init_normal

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

        # Initialize instance variables
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed

        # Q-Networks
        self.qnetworks_local = QNetworks(state_size, action_size, seed).to(device)
        self.qnetworks_local.apply(weights_init_normal)
        self.qnetworks_target = QNetworks(state_size, action_size, seed).to(device)
        self.qnetworks_target.apply(weights_init_normal)

        # Freeze gradients for parameters of qnetworks_target
        # that are not used for the learning.
        for param in self.qnetworks_target.parameters():
            param.requires_grad_(False)

        # Initialize elements for the training
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.qnetworks_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

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