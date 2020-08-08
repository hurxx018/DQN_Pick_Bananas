import os

from unityagents import UnityEnvironment

import random

import torch

import numpy as np
import matplotlib.pyplot as plt


from dqn_agent import DQNAgent
from train import train

def train_agent_banana(
    filename_unity_environment,
    ):

    # Open the environment
    env = UnityEnvironment(file_name = filename_unity_environment, no_graphics=True)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    agent = DQNAgent(state_size=state_size , action_size=action_size, seed=None)
    print()
    print(agent.qnetworks_local)

    scores = train(env, agent, brain_name, n_episodes = 2000, eps_start = 0.90, eps_end = 0.0001, eps_decay = 0.99)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(os.path.join(".", "images", "score.png"))
    plt.show()

    return 


def test_agent_banana(
    filename_unity_environment,
    no_graphics = False
    ):

    env = UnityEnvironment(file_name = filename_unity_environment, no_graphics = no_graphics)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=False)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)
    print()
    agent = DQNAgent(state_size=state_size , action_size=action_size, seed=None)

    # read weights
    agent.qnetworks_local.load_state_dict(torch.load('checkpoint.pth'))

    # run the agent
    n_episodes = 3
    epsilon = 0.005
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=False)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0                                          # initialize the score
        while True:
            action = agent.act(state, epsilon)        # select an action
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished

            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            if done:                                       # exit loop if episode finished
                break

        print(score)

    return


def main():
    # a path for Banana app.
    filename_unity_environment = "..."
    
    # train the agent
    # if you are interested in watching the result of the trained agent,
    # comment out the next line.
    train_agent_banana(filename_unity_environment)

    # see the agent's work
    no_graphics = False # control the display of unit environment
    test_agent_banana(filename_unity_environment, no_graphics = no_graphics)




if __name__ == "__main__":
    main()