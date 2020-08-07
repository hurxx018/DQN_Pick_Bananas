
from collections import deque

import numpy as np

import torch

def train(
    env,
    agent, 
    brain_name,
    n_episodes=2000,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.99
    ):

    scores = [] # list of scores
    scores_window = deque(maxlen=100)  # last 100 scores

    eps = eps_start
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0                                          # initialize the score
        while True:
            action = agent.act(state, eps)        # select an action
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished

            agent.step(state, action, reward, next_state, done) # Update the value function
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            if done:                                       # exit loop if episode finished
                break

        scores.append(score)
        scores_window.append(score)
        eps = max(eps*eps_decay, eps_end)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

        # Save the model if the mean of 100 continuing scores is greater than 13.
        m = np.mean(scores_window)
        if m > 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, m))
            torch.save(agent.qnetworks_local.state_dict(), 'checkpoint.pth')
            break
    return scores