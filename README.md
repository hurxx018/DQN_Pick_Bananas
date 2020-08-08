# Deep Q-Networks for an agent that picks up banana

## Problem
An agent navigates a field where yellow and blue bananas are located randomly. The agent receives a reward of +1 whenever it picks a yellow banana up and a reward of -1 for collecting a blue banana. The goal is to train the agent to be able to receive on average total rewards of at least 13 during 100 consecutive episodes. The state space is continuous and it consists of 37 degrees of freedom including the agent's velocity along with ray-based perception of objects around the agent's forward direction. There are 4 choices of discrete action. 

## Getting Started
To install the UnityEnvironment, follow the instruction in [Here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation)

## Instructions
To train the agent and see the quality of the trained agent, run the following after putting the filename of unity environment.

python ./main.py

The weight of the agent is stored in checkpoint.pth.

If you are interested in only watching the play of the trained agent, comment out the line of training step (e.g. #train_agent_banana(filename_unity_environment))

## Report
[Here is a report](Report.md)
