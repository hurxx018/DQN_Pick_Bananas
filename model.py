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
        state_size,
        action_size,
        seed
        ):
        super(QNetworks, self).__init__()



    def forward(
        self, 
        x
        ):
        pass