import sys
print(sys.path)


from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn

import gymnasium as gym
from agent import AcrobotDQNAgent

writer = SummaryWriter('runs/Acrobot_1')

def train_agent():
    # 初始化环境和智能体
    env = gym.make('Acrobot-v1',render_mode="rgb_array")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = AcrobotDQNAgent(state_size, action_size,epsilon_decay=0.9999,batch_size=128,device="cuda:2")

    num_episodes = 1000000
    max_timesteps = 500
    target_update_frequency = 100

    for episode in range(num_episodes):
        state,info = env.reset()
        total_reward = 0
        total_loss = 0
        total_target_q_values = 0
        for t in range(max_timesteps):

            action = agent.choose_action(state)
            next_state, reward, terminated , truncated, info  = env.step(action)
            
            agent.remember(state, action, reward, next_state, terminated)
            loss,target_q_values = agent.train()

            total_loss += loss
            # total_target_q_values += target_q_values
            state = next_state
            total_reward += reward

            if terminated:
                break
        writer.add_scalar('reward', total_reward, episode)
        writer.add_scalar('loss', total_loss, episode)
        writer.add_histogram('target_q_values', total_target_q_values, episode)
        # 每100个回合更新一次目标网络
        if episode % target_update_frequency == 0:
            agent.update_target_model()

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

        # 保存模型
        if episode % 100 == 0:
            agent.save_model(f"acrobot_dqn_{episode}.pth")


if __name__=="__main__":
    train_agent()