# Report

## Implementation
The agent utilizes a deep Q-network as an approximation of a value function that consists of two hidden layers (see the section of Architecture for the details). The agent includes two deep Q networks. The one labeled by local is used to learn the value function and the other labeled by target plays a ker role of approximating the true value function of the environment. During the training, the target Q network is updated periodically by employing the experiences stored in the replay buffer or the agent's memory. The periodicity is defined by a parameter UPDATE_EVERY in dqn_agent.py. The number of experiences used for the learning is defined by BATCH_SIZE in dqn_agent.py. In this work, BATCH_size is 128. The agent employs the epsilon-greedy policy to handle exploitation-exploration dilemma. The initial value of epsilon is given by 0.9 in this implementation and it becomes discounted for running each next episode by a rate of 0.99.

## Learning Algorithm
The agent utilizes the Deep Q-Networks with an epsilon-greedy policy. The replay buffer is

## Architecture
The Network consists of an input layer (black), two hidden layers (blue), and an output layer (red). Only fully connected layers denoted by fc1 and fc2 are used with a rectified linear unit (ReLU) activation. Each hidden layer includes 128 nodes.

![figure of architecture](https://github.com/hurxx018/DQN_Pick_Bananas/blob/master/images/architecture_for_agent.png)


## Results
This is a trace of mean values of 100 consecutive scores. The agent achieves the goal after running ~1450 episodes. The initial stage of learning is steep and the learning phase becomes slow down. 

![figure of score](https://github.com/hurxx018/DQN_Pick_Bananas/blob/master/images/score.png)

It would be easy to make a decision of picking the first five pieces of bananas. After doing it, the agent might avoid blue bananas left around itself and move toward picking yellow bananas in the other area. These two statements are not proven yet here. The next question could be "How can we show the process that the agent learns the strategy?". 

## Future Work
This work needs to be improved with the implementation of prioritized experience replay and dueling Q-networks.