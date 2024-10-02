# src/drl.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class D3QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(D3QNetwork, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU()
        )
        # Value Stream
        self.value_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # Advantage Stream
        self.advantage_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_layer(features)
        advantages = self.advantage_layer(features)
        qvals = values + (advantages - advantages.mean())
        return qvals

class D3PAgent:
    def __init__(self, state_size, action_size, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.99
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.model = D3QNetwork(state_size, action_size)
        self.target_model = D3QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def memorize(self, state, action, reward, next_state, done, error):
        self.memory.append((state, action, reward, next_state, done, error))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state)
        q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        # Prioritized Experience Replay based on TD error
        minibatch = random.sample(self.memory, batch_size)
        states = []
        targets = []
        for state, action, reward, next_state, done, error in minibatch:
            state_tensor = torch.FloatTensor(state)
            next_state_tensor = torch.FloatTensor(next_state)
            target = self.model(state_tensor).detach()
            if done:
                target[action] = reward
            else:
                t = self.target_model(next_state_tensor).detach()
                target[action] = reward + self.gamma * torch.max(t)
            states.append(state_tensor)
            targets.append(target)

        states_tensor = torch.stack(states)
        targets_tensor = torch.stack(targets)

        self.optimizer.zero_grad()
        outputs = self.model(states_tensor)
        loss = self.criterion(outputs, targets_tensor)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
