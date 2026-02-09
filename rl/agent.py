import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from rl.dqn import DQN

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.lr = 1e-3
        self.batch_size = 64

        self.memory = deque(maxlen=50000)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(state_size, action_size).to(self.device)
        self.target = DQN(state_size, action_size).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.step_count = 0

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        state = torch.tensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def remember(self, s, a, r, s_next):
        self.memory.append((s, a, r, s_next))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.from_numpy(np.array(states)).float().to(self.device)
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)

        q = self.model(states).gather(1, actions).squeeze()
        q_next = self.target(next_states).max(1)[0].detach()
        target = rewards + self.gamma * q_next

        loss = self.loss_fn(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.step_count += 1
        if self.step_count % 1000 == 0:
            self.target.load_state_dict(self.model.state_dict())

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
   
    def load(self, path):
        self.model.load_state_dict(
        torch.load(path, map_location=self.device)
        )
        self.model.eval()