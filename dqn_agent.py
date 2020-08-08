import random
from collections import deque, namedtuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import QNetworks, weights_init_normal

BUFFER_SIZE = int(2e5)  # replay buffer size
BATCH_SIZE = 64*2         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-3               # learning rate 
UPDATE_EVERY = 4*1        # how often to update the network

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
        self.seed = random.seed(seed)
        np.random.seed(seed)
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
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(
        self,
        state,
        epsilon = 0. 
        ):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.qnetworks_local.eval()
        with torch.no_grad():
            action_values = self.qnetworks_local(state)
        self.qnetworks_local.train()

        # epsilon-greedy policy
        a = torch.argmax(action_values)
        policy = np.full(self.action_size, epsilon/self.action_size)
        policy[a] += 1. - epsilon

        return np.random.choice(np.arange(self.action_size), p = policy)

    def learn(
        self,
        experiences,
        gamma
        ):
        """ """
        states, actions, rewards, next_states, dones = experiences

        self.optimizer.zero_grad()

        outputs = self.qnetworks_local(states)
        outputs = outputs.gather(1, actions)
        z = self.qnetworks_target(next_states)
        max_values, _ = torch.max(z, dim = 1)
        targets = rewards + gamma*(max_values.unsqueeze(1)*(1. - dones))

        loss = self.criterion(outputs, targets)
        # Minimize the loss
        loss.backward()
        self.optimizer.step()
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetworks_local, self.qnetworks_target, TAU)

    def soft_update(
        self,
        local_model,
        target_model,
        tau
        ):
        """ """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer(object):
    """ ReplayBuffer
        Fixed-size buffer to store experience tuples.
    """
    def __init__(
        self,
        action_size,
        buffer_size,
        batch_size, 
        seed
        ):
        """ Initialize a ReplayBuffer object
            Arguments
            ---------
            action_size : int
                Number of actions for an agent
            buffer_size : int
                Maximum size of buffer
            batch_size : int
                Number of experiences for each batch
            seed : int
                random seed
        """
        # Store instance variables
        self.action_size = action_size
        self.memory = deque(maxlen = buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)

        # Namedtuple for storing each experience
        self._experience = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"])

    def add(
        self,
        state,
        action, 
        reward,
        next_state, 
        done
        ):
        """ Add a new experience to memory """
        tmp_e = self._experience(state, action, reward, next_state, done)
        self.memory.append(tmp_e)

    def sample(
        self
        ):
        """ Randomly sample a batch of experiences from memory """
        # randomly chosen experiences
        experiences = random.sample(self.memory, k = self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(
        self
        ):
        """ Return the current size of internal memory. """
        return len(self.memory)