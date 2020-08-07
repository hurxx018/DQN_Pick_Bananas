# Deep Q-Networks for an agent that picks up banana

## Problem
An agent navigates a field where yellow and blue bananas are located randomly. The agent receives +1 as a reward whenever it picks a yellow banana up and -1 for a blue banana. The goal is to train the agent to be able to receive on average total rewards of at least 13 during continuing 100 episodes. The state space is continuous and it consists of 37 degrees of freedom. There are 4 choices of action. 

## Solution
The agent utilizes a deep Q-network as value function that consists of hidden layers (see the architecture for the details). The agent includes two deep Q networks. The one labeled by local is used to learn the value function and the other labeled by target plays a ker role of approximating the true value function of the environment. During the training, the target Q network is updated periodically. The periodicity is defined by a parameter UPDATE_EVERY in dqn_agent.py. The agent employs the epsilon-greedy policy to handle exploitation-exploration dilemma. The initial value of epsilon is given by 0.9 and it becomes discounted each episode.

## Instruction
QNetworks in model.py
Agent in dqn_agent.py
train in train.py




## Architecture
image


## Result
![figure of score]("score.png")