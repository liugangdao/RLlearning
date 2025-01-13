from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

class AcrobotDQNAgent:
  def __init__(self,state_size,action_size,buffer_size=10000,batch_size=64, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.001,device="cpu"):
    self.device = torch.device(device)
    self.state_size = state_size
    self.action_size = action_size

    self.buffer_size = buffer_size
    self.batch_size = batch_size
    self.gamma = gamma
    self.epsilon = epsilon
    self.epsilon_decay = epsilon_decay
    self.epsilon_min = epsilon_min

    self.memory = ReplayBuffer(buffer_size)
    self.model = DQNNetwork(state_size, action_size).to(self.device)
    self.target_model = DQNNetwork(state_size, action_size).to(self.device)
    self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    self.loss_fn = nn.MSELoss()

    self.update_target_freq = 1000

  def remember(self, state, action, reward, next_state, done):
    self.memory.add((state, action, reward, next_state, done))

  def update_target_model(self):
    self.target_model.load_state_dict(self.model.state_dict())

  def choose_action(self, state):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

    if np.random.rand() <= self.epsilon:
      return np.random.choice(self.action_size)
    else:
      with torch.no_grad():
        q_values = self.model(state)
        action_probs = torch.softmax(q_values, dim=-1)
        return torch.argmax(action_probs).item()

  def train(self):
    if self.memory.size() < self.batch_size:
      return 0, 0

    batch = self.memory.sample(self.batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32).to(self.device)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

    with torch.no_grad():
      next_q_values = self.target_model(next_states)
      max_next_q_values, _ = torch.max(next_q_values, dim=1, keepdim=True)
      target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

    q_values = self.model(states).gather(1, actions)
    loss = self.loss_fn(q_values, target_q_values)

    # 优化
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    # 更新 ε
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay
    
    return loss.item(),target_q_values.mean().item()
  
  def save_model(self, path):
    torch.save(self.model.state_dict(), path)

  def load_model(self, path):
    self.model.load_state_dict(torch.load(path))