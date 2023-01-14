import gym
import numpy as np
import torch
from torch import nn
import pdb

import utils
from policies import QPolicy

# Modified by Mohit Goyal (mohit@illinois.edu) on 04/20/2022

def make_dqn(statesize, actionsize):
    """
    Create a nn.Module instance for the q leanring model.

    @param statesize: dimension of the input continuous state space.
    @param actionsize: dimension of the descrete action space.

    @return model: nn.Module instance
    """
    return nn.Sequential(nn.Linear(statesize, 32), nn.ReLU(), nn.Linear(32,32), nn.ReLU(), nn.Linear(32, actionsize))


class DQNPolicy(QPolicy):
    """
    Function approximation via a deep network
    """

    def __init__(self, model, statesize, actionsize, lr, gamma):
        """
        Inititalize the dqn policy

        @param model: the nn.Module instance returned by make_dqn
        @param statesize: dimension of the input continuous state space.
        @param actionsize: dimension of the descrete action space.
        @param lr: learning rate 
        @param gamma: discount factor
        """
        super().__init__(statesize, actionsize, lr, gamma)
        if model is None:
            self.model = make_dqn(statesize, actionsize)
        else:
            self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=lr)
        self.loss = nn.MSELoss()
        self.lr = lr
        self.gamma = gamma

    def qvals(self, state):
        """
        Returns the q values for the states.

        @param state: the state
        
        @return qvals: the q values for the state for each action. 
        """
        self.model.eval()
        with torch.no_grad():
            states = torch.from_numpy(state).type(torch.FloatTensor)
            qvals = self.model(states)
        return qvals.numpy()

    def td_step(self, state, action, reward, next_state, done):
        """
        One step TD update to the model

        @param state: the current state
        @param action: the action
        @param reward: the reward of taking the action at the current state
        @param next_state: the next state after taking the action at the
            current state
        @param done: true if episode has terminated, false otherwise
        @return loss: total loss the at this time step
        """
        cur_states = torch.from_numpy(state).type(torch.FloatTensor)
        
        if done:
            target = torch.tensor(reward)
        else:
            next_states = torch.from_numpy(next_state).type(torch.FloatTensor)
            next_qval = torch.max(self.model(next_states))
            target = reward + self.gamma*next_qval

        loss = self.loss((self.model(cur_states))[action], target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self, outpath):
        """
        saves the model at the specified outpath
        """        
        torch.save(self.model, outpath)


if __name__ == '__main__':
    args = utils.hyperparameters()

    env = gym.make('CartPole-v1')
    env.reset(seed=42) # seed the environment
    np.random.seed(42) # seed numpy
    import random
    random.seed(42)
    torch.manual_seed(0) # seed torch
    torch.use_deterministic_algorithms(True) # use deterministic algorithms

    statesize = env.observation_space.shape[0]
    actionsize = env.action_space.n

    policy = DQNPolicy(make_dqn(statesize, actionsize), statesize, actionsize, lr=args.lr, gamma=args.gamma)

    utils.qlearn(env, policy, args)

    torch.save(policy.model, 'dqn.model')
