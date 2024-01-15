import argparse
import gym
import numpy as np
from itertools import count
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import pygame
import sys
import time
from signals import SignalGenerator
from environment import Environment


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.999, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

pygame.init()
screen_width = 1024
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()

signal_generator = SignalGenerator(screen)
env = Environment(screen=screen, render_on=True, signal_generator=signal_generator)
torch.manual_seed(args.seed)
env.reset()


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        #self.affine1 = nn.Linear(65536*2, 2048)
        self.affine1 = nn.Linear(4096, 2048)
        self.affine2 = nn.Linear(2048, 2048)
        self.affine3 = nn.Linear(2048, 1024)
        # self.dropout = nn.Dropout(p=0.6)
        self.affineS = nn.Linear(1024, 1024)
        self.affineE = nn.Linear(1024, 1024)
        self.saved_log_probs = []
        self.saved_log_probe = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        # x = self.dropout(x)
        x = F.relu(x)
        x = self.affine2(x)
        x = F.relu(x)
        x = self.affine3(x)
        x = F.relu(x)
        start_scores = self.affineS(x)
        end_scores = self.affineE(x)
        return F.softmax(start_scores, dim=1),F.softmax(end_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = np.abs(state.astype(np.float32).view(np.complex64))
    state = state.reshape((-1,16))
    state = np.mean(state, axis=1)
    print("min",np.min(state),"max",np.max(state))
    state = torch.from_numpy(state).float().unsqueeze(0)
    sprobs,eprobs = policy(state)
    m = Categorical(sprobs)
    n = Categorical(eprobs)
    actions = m.sample()
    actione = n.sample()

    policy.saved_log_probs.append(sprobs)
    policy.saved_log_probe.append(eprobs)
    return actions, actione


def finish_episode():
    R = 0
    policy_loss = []
    returns = deque()
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.appendleft(R)
    if len(returns) < 10:
        return
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    
    for log_probs, log_probe, R in zip(policy.saved_log_probs,policy.saved_log_probe, returns):
        policy_loss.append(-log_probs * R + -log_probe * R)
    optimizer.zero_grad()
    #print("policy loss",policy_loss)
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]
    del policy.saved_log_probe[:]


def main():
    running_reward = 10
    for i_episode in count(1):
        state = env.reset()
        ep_reward = 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            actions,actione = select_action(state)
            reward, state, done = env.step(actions,actione)
            #print(state, reward)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if False: #running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
