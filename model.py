import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetworks(nn.Module):
    """ Actor Model.
        This model is a function approximation of action-values. 
        The networks is a mapping from a state to a list of action values, so
        the agent uses its own policy such as greedy- or epsilon-greedy policies.
        The model is implemented as a fully-connected multilayer networks. 


        Arguments
        ---------
        state_size : int
            Number of states of the environment.
        action_size : int
            Number of actions that the agent can take.
        seed : int
            Seed for a random number generator.

    """
    def __init__(
        self,
        state_size = 37,
        action_size = 4,
        seed = 12345
        ):
        super(QNetworks, self).__init__()

        # create instance variables
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed

        # compute 2**n closest to state_size
        e2 = int(2**int(math.log2(self.state_size)))

        # create layers
        self.fc1 = nn.Linear(self.state_size, e2*2)
        self.fc2 = nn.Linear(e2*2, e2*4)
        self.fc3 = nn.Linear(e2*4, e2*2)
        self.fc4 = nn.Linear(e2*2, e2*1)
        self.fc5 = nn.Linear(e2*1, self.action_size)

    def forward(
        self, 
        x
        ):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        out = F.relu(self.fc5(x))

        return out


def weights_init_normal(
    m
    ):
    """ Initialize a linear layer 
        Arguments
        ---------
        m : 
            a layer of model
    """
    if isinstance(m, nn.Linear):
        n = m.in_features
        y = 1./math.sqrt(n)
        m.weight.data.normal_(0., y)
        m.bias.data.fill_(0)
