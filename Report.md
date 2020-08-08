# Report

## Implementation


## Learning Algorithm
Deep Q-Networks with an epsilon-greedy policy.

## Architecture
The Network consists of an input layer (black), two hidden layers (blue), and an output layer (red). Only fully connected layers are used with a rectified linear unit (ReLU) activation.
![figure of architecture](https://github.com/hurxx018/DQN_Pick_Bananas/blob/master/images/architecture_for_agent.png)


## Results
This is a trace of mean values of 100 consecutive scores. The agent achieves the goal after running ~1450 episodes. The initial stage of learning is steep and the learning phase becomes slow down. 
![figure of score](https://github.com/hurxx018/DQN_Pick_Bananas/blob/master/images/score.png)
It would be easy to make a decision of picking the first five pieces of bananas. After doing it, the agent might avoid blue bananas left around itself and move toward picking yellow bananas in the other area. These two statements are not proven yet here. The next question could be "How can we show the process that the agent learns the strategy?". 

## Future Work