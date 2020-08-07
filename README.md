# Deep Q-Networks for an agent that picks up banana

## Problem
An agent navigates a field where yellow and blue bananas are located randomly. The agent receives +1 as a reward whenever it picks a yellow banana up and -1 for a blue banana. The goal is to train the agent to be able to receive on average total rewards of at least 13 during continuing 100 episodes. The state space is continuous and it consists of 37 degrees of freedom. There are 4 choices of action. 

## Solution
The agent utilizes a deep Q-network as value function that consists of hidden layers (see the architecture for the details). 


## Architecture
image