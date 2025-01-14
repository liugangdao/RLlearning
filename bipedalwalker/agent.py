"""
对于连续性的动作空间，采用Deep Deterministic Policy Gradient（确定性策略梯度）是比较适用的
"""

from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim




class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    def __init__(self, state_size,action_size,hidden_dim=256):
        super(Critic,self).__init__()
        self.fc1 =  nn.Linear(state_size, hidden_dim)
        self.fc2 =  nn.Linear(action_size, hidden_dim)
        self.fc3 =  nn.Linear(hidden_dim*2,hidden_dim)
        self.fc4 =  nn.Linear(hidden_dim, 1)
    
    def forward(self,state,action):
        x1 = torch.relu(self.fc1(state))
        x2 = torch.relu(self.fc2(action))
        x = torch.cat([x1,x2],dim=1)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)
 
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

# 在Actor网络输出的动作上添加OU噪声
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.action_dim = action_dim
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state = self.state + dx
        return self.state

class DDPG:

    def __init__(self,state_size,action_size,
                 device="cpu",
                 buffer_size=100000, 
                 tau=0.005,
                 batch_size=64, 
                 gamma=0.99, 
                 mu=1.0, 
                 theta=0.995, 
                 sigma=0.01, 
                 epsilon=1.0, 
                 epsilon_decay=0.995, 
                 epsilon_min=0.01,
                 lr_actor=0.001,
                 lr_critic=0.001,
                 update_target_frequency=100):
        self.state_size = state_size
        self.action_size = action_size
        self.tau = tau

        self.device = torch.device(device)
        self.actor = Actor(state_size,action_size).to(self.device)
        self.target_actor = Actor(state_size,action_size).to(self.device)
        self.critic = Critic(state_size,action_size).to(self.device)
        self.target_critic = Critic(state_size,action_size).to(self.device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(),lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(),lr=lr_critic)
        self.update_target_network(tau=1)

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

        self.gamma = gamma
        
        self.noise = OUNoise(action_dim=action_size,mu=mu,theta=theta,sigma=sigma)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.update_target_frequency = update_target_frequency

    def update_target_network(self,tau=None):
        tau = tau or self.tau
        for target_parames, params in zip(self.target_actor.parameters(),self.actor.parameters()):
            target_parames.data.copy_((1-tau)*target_parames.data + tau*params.data)
        for target_parames, params in zip(self.target_critic.parameters(),self.target_critic.parameters()):
            target_parames.data.copy_((1-tau)*target_parames.data + tau*params.data)

    def choose_action(self,state):
        state = torch.tensor(state,dtype=torch.float32).to(self.device)
        action = self.actor(state).detach().to('cpu').numpy()

        # action = np.clip(action + noise_scale * np.random.randn(self.action_dim), -self.action_bound, self.action_bound)
        action = np.clip(action + self.noise.sample()*max(self.epsilon,self.epsilon_min))
        return action
    
    def train(self):
        if self.replay_buffer.size()<self.batch_size:
            return 0

        # 从经验回放缓冲区采样
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1).to(self.device)

        # 计算目标Q值
        next_action = self.target_actor(next_states)
        next_q_value = self.target_critic(next_states,next_action)
        target_q_values = rewards + (1 - dones)*self.gamma*next_q_value
        # target_q_values = rewards
        
        # 更新critic网络
        current_q_values = self.critic(states,actions)
        critic_loss = torch.mean((current_q_values-target_q_values.detach())**2)
        
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # 更新Actor网络
        actor_loss = -torch.mean(self.critic(states,self.actor(states)))
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # 更新目标网络
        self.update_target_network()
        
        # 更新 ε
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return actor_loss.item()
    
    def save_model(self,episode, filename):
        # 保存Actor网络和Critic网络的参数
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.target_actor.state_dict(),
            'critic_target_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.optimizer_actor.state_dict(),
            'critic_optimizer_state_dict': self.optimizer_critic.state_dict(),
            'episode': episode  # 保存当前的episode，用于恢复训练
        }, filename)
        print(f"{episode} Model saved to {filename}")
    
    def load_model(self, filename):
        checkpoint = torch.load(filename)
        
        # 恢复Actor和Critic网络的参数
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_actor.load_state_dict(checkpoint['actor_target_state_dict'])
        self.target_critic.load_state_dict(checkpoint['critic_target_state_dict'])
        
        

        # 恢复优化器的状态
        self.optimizer_actor.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.optimizer_critic.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        # 恢复训练的进度（例如从哪个episode开始）
        self.episode = checkpoint['episode']
        
        print(f"Model loaded from {filename}, starting from episode {self.episode}")