import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.7, n_step=5, gamma=0.95):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = 0.3
        self.beta_increment = 0.0001  # Adjust as needed
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)

    def add(self, td_error, experience):
        self.n_step_buffer.append(experience)
        if len(self.n_step_buffer) < self.n_step:
            return
        reward, next_state, done = self._get_n_step_info()
        state, action = self.n_step_buffer[0][:2]
        n_step_experience = (state, action, reward, next_state, done)

        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(n_step_experience)
        else:
            self.buffer[self.pos] = n_step_experience
        self.priorities[self.pos] = max(abs(td_error), max_prio) ** self.alpha
        self.pos = (self.pos + 1) % self.capacity

    def _get_n_step_info(self):
        reward, next_state, done = self.n_step_buffer[-1][2:]
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, next_s, d = transition[2:]
            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (next_s, d) if d else (next_state, done)
        return reward, next_state, done

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]
        probs = prios / prios.sum()
        self.beta = min(1.0, self.beta + self.beta_increment)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return experiences, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) ** self.alpha

class D3PAgent:
    def __init__(self, state_size, action_size, lr=1e-5):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(capacity=2000, alpha=0.7)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.0001
        self.update_period = 5
        self.steps = 0

        self.model = D3QNetwork(state_size, action_size)
        self.target_model = D3QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        self.criterion = nn.SmoothL1Loss()

        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        """
        Epsilon-greedy action selection.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
            q_values = self.model(state)
            return torch.argmax(q_values, dim=1).item()

    def memorize(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state)
        next_state_tensor = torch.FloatTensor(next_state)
        current_q = self.model(state_tensor.unsqueeze(0))[0][action].item()
        next_q = reward
        if not done:
            next_action = torch.argmax(self.model(next_state_tensor.unsqueeze(0)))
            next_q += self.gamma * self.target_model(next_state_tensor.unsqueeze(0))[0][next_action].item()
        td_error = current_q - next_q
        self.memory.add(td_error, (state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory.buffer) < batch_size:
            return  # Not enough samples to replay

        experiences, indices, weights = self.memory.sample(batch_size)
        batch = list(zip(*experiences))
        states = torch.FloatTensor(np.array(batch[0]))
        actions = torch.LongTensor(batch[1])
        rewards = torch.FloatTensor(batch[2])
        next_states = torch.FloatTensor(np.array(batch[3]))
        dones = torch.FloatTensor(batch[4])
        weights_tensor = torch.FloatTensor(weights)

        # Current Q Values
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN - Next Actions from Online Model
        next_actions = self.model(next_states).argmax(1)
        # Next Q Values from Target Model
        next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        # Compute Target Q Values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        td_errors = q_values.detach() - target_q_values.detach()
        self.memory.update_priorities(indices, td_errors.numpy())

        # Compute Loss
        loss = (weights_tensor * self.criterion(q_values, target_q_values.detach())).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        # Update target network periodically
        self.steps += 1
        if self.steps % self.update_period == 0:
            self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
