import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# 定义网络结构
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 定义策略网络（Actor）
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_mean = nn.Linear(hidden_dim, action_dim)
        self.fc3_logstd = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.fc3_mean(x)
        log_std = self.fc3_logstd(x)
        
        # 限制log_std的范围，防止极端值
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        
        return mean, std

    def sample(self, state):
        mean, std = self(state)
        
        # 防止标准差为零
        std = torch.clamp(std, min=1e-6)
        
        normal_dist = torch.distributions.Normal(mean, std)
        action = normal_dist.rsample()  # Reparameterized action
        log_prob = normal_dist.log_prob(action).sum(dim=-1, keepdim=True)  # logP(a|s)
        
        # 限制log_prob的范围
        log_prob = torch.clamp(log_prob, min=-10, max=10)
        
        return action, log_prob

# 定义Q网络（Critic）
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# SAC算法类
class SAC:
    def __init__(self, state_dim, action_dim, action_high, action_low, gamma=0.99, tau=0.05, alpha=0.2):
        self.gamma = gamma  # Discount factor
        self.tau = tau  # Target update rate
        self.alpha = alpha  # Entropy weight

        # 定义网络
        self.actor = Actor(state_dim, action_dim).cuda()
        self.critic1 = QNetwork(state_dim, action_dim).cuda()
        self.critic2 = QNetwork(state_dim, action_dim).cuda()
        self.target_critic1 = QNetwork(state_dim, action_dim).cuda()
        self.target_critic2 = QNetwork(state_dim, action_dim).cuda()

        # 初始化目标网络
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-3)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-3)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-3)

        # 动作范围
        self.action_high = action_high
        self.action_low = action_low

    def select_action(self, state):
        # print(state)
        state = torch.tensor(state,dtype=torch.float32).unsqueeze(0).cuda()
        action, _ = self.actor.sample(state)
        action = torch.tanh(action)  # 将动作限制在 [-1, 1] 范围内
        action = action * (self.action_high - self.action_low) / 2 + (self.action_high + self.action_low) / 2  # 恢复原始范围
        return action.cpu().detach().numpy()[0]

    def update(self, replay_buffer, batch_size=256):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.tensor(state,dtype=torch.float32).cuda()
        action = torch.tensor(action,dtype=torch.float32).cuda()
        reward = torch.tensor(reward,dtype=torch.float32).cuda().unsqueeze(-1)
        next_state = torch.tensor(next_state,dtype=torch.float32).cuda()
        done = torch.tensor(done,dtype=torch.float32).cuda().unsqueeze(-1)

        # 计算目标Q值
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            next_action = torch.tanh(next_action)  # 保持动作在 [-1, 1] 范围内

            target_q1 = self.target_critic1(next_state, next_action)
            target_q2 = self.target_critic2(next_state, next_action)

            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target = reward + (1 - done) * self.gamma * target_q

        # 计算当前Q值
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)

        # 计算Q网络损失并更新
        critic1_loss = torch.mean((current_q1 - target) ** 2)
        critic2_loss = torch.mean((current_q2 - target) ** 2)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # 更新Actor网络
        action, log_prob = self.actor.sample(state)
        action = torch.tanh(action)
        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        q = torch.min(q1, q2)

        actor_loss = (self.alpha * log_prob - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新目标网络
        self.soft_update(self.target_critic1, self.critic1)
        self.soft_update(self.target_critic2, self.critic2)
        return actor_loss

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save_model(self,episode, filename):
        # 保存Actor网络和Critic网络的参数
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_1_state_dict': self.target_critic1.state_dict(),
            'critic_2_state_dict': self.target_critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic_2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'episode': episode  # 保存当前的episode，用于恢复训练
        }, filename)
        print(f"{episode} Model saved to {filename}")
    

# Replay Buffer类
class ReplayBuffer:
    def __init__(self, buffer_size=1e6):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=int(buffer_size))

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)

    def size(self):
        return len(self.buffer)


# 训练过程
def train():
    env = gym.make("BipedalWalker-v3",hardcore=True, render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high = env.action_space.high[0]
    action_low = env.action_space.low[0]

    sac = SAC(state_dim, action_dim, action_high, action_low)
    replay_buffer = ReplayBuffer()

    episodes = 1000
    batch_size = 256
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0

        while True:
            action = sac.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            replay_buffer.add((state, action, reward, next_state, done))

            if replay_buffer.size() > batch_size:
                sac.update(replay_buffer, batch_size)

            state = next_state
            episode_reward += reward

            if done:
                break

        print(f'Episode {episode+1}/{episodes}, Reward: {episode_reward}')

if __name__ == "__main__":
    train()