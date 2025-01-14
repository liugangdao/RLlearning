from collections import defaultdict
import gymnasium as gym
import numpy as np

from sac import SAC, ReplayBuffer

from agent import DDPG

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/bipedalwarker_sac')

def test_env():
    env = gym.make("BipedalWalker-v3",hardcore=True, render_mode="rgb_array")

    state, info = env.reset()

    # print(state)

    # for i in range(2001):
    #     action = env.action_space.sample()
    #     print(action)

    #     next_obs, reward, terminated, truncated,info  = env.step(action)

    #     print(f"next_obs : {next_obs}")
    #     print(f"reward : {reward}")
    #     print(f"terminated : {terminated}")
    #     print(f"truncated : {truncated}")
    #     print(f"info: {info}")
    #     if terminated and truncated:
    #         print("step: ",i)
    #         break
    steps = 0
    total_reward = 0
    a = np.array([0.0, 0.0, 0.0, 0.0])
    STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1, 2, 3
    SPEED = 0.29  # Will fall forward on higher speed
    state = STAY_ON_ONE_LEG
    moving_leg = 0
    supporting_leg = 1 - moving_leg
    SUPPORT_KNEE_ANGLE = +0.1
    supporting_knee_angle = SUPPORT_KNEE_ANGLE
    while True:
        s, r, terminated, truncated, info = env.step(a)
        total_reward += r
        if steps % 20 == 0 or terminated or truncated:
            print("\naction " + str([f"{x:+0.2f}" for x in a]))
            print(f"step {steps} total_reward {total_reward:+0.2f}")
            print("hull " + str([f"{x:+0.2f}" for x in s[0:4]]))
            print("leg0 " + str([f"{x:+0.2f}" for x in s[4:9]]))
            print("leg1 " + str([f"{x:+0.2f}" for x in s[9:14]]))
            print(f"terminated : {terminated}")
            print(f"truncated : {truncated}")
        steps += 1

        contact0 = s[8]
        contact1 = s[13]
        moving_s_base = 4 + 5 * moving_leg
        supporting_s_base = 4 + 5 * supporting_leg

        hip_targ = [None, None]  # -0.8 .. +1.1
        knee_targ = [None, None]  # -0.6 .. +0.9
        hip_todo = [0.0, 0.0]
        knee_todo = [0.0, 0.0]

        if state == STAY_ON_ONE_LEG:
            hip_targ[moving_leg] = 1.1
            knee_targ[moving_leg] = -0.6
            supporting_knee_angle += 0.03
            if s[2] > SPEED:
                supporting_knee_angle += 0.03
            supporting_knee_angle = min(supporting_knee_angle, SUPPORT_KNEE_ANGLE)
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[supporting_s_base + 0] < 0.10:  # supporting leg is behind
                state = PUT_OTHER_DOWN
        if state == PUT_OTHER_DOWN:
            hip_targ[moving_leg] = +0.1
            knee_targ[moving_leg] = SUPPORT_KNEE_ANGLE
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[moving_s_base + 4]:
                state = PUSH_OFF
                supporting_knee_angle = min(s[moving_s_base + 2], SUPPORT_KNEE_ANGLE)
        if state == PUSH_OFF:
            knee_targ[moving_leg] = supporting_knee_angle
            knee_targ[supporting_leg] = +1.0
            if s[supporting_s_base + 2] > 0.88 or s[2] > 1.2 * SPEED:
                state = STAY_ON_ONE_LEG
                moving_leg = 1 - moving_leg
                supporting_leg = 1 - moving_leg

        if hip_targ[0]:
            hip_todo[0] = 0.9 * (hip_targ[0] - s[4]) - 0.25 * s[5]
        if hip_targ[1]:
            hip_todo[1] = 0.9 * (hip_targ[1] - s[9]) - 0.25 * s[10]
        if knee_targ[0]:
            knee_todo[0] = 4.0 * (knee_targ[0] - s[6]) - 0.25 * s[7]
        if knee_targ[1]:
            knee_todo[1] = 4.0 * (knee_targ[1] - s[11]) - 0.25 * s[12]

        hip_todo[0] -= 0.9 * (0 - s[0]) - 1.5 * s[1]  # PID to keep head strait
        hip_todo[1] -= 0.9 * (0 - s[0]) - 1.5 * s[1]
        knee_todo[0] -= 15.0 * s[3]  # vertical speed, to damp oscillations
        knee_todo[1] -= 15.0 * s[3]

        a[0] = hip_todo[0]
        a[1] = knee_todo[0]
        a[2] = hip_todo[1]
        a[3] = knee_todo[1]
        a = np.clip(0.5 * a, -1.0, 1.0)

        if terminated or truncated:
            break

def train():
    env = gym.make("BipedalWalker-v3",hardcore=True, render_mode="rgb_array")
    device = "cuda:2"
    episodes=10000
    max_timesteps=2000
    save_freq = 100
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent_ddpg = DDPG(state_dim,action_dim,device=device,batch_size=128,epsilon=1,epsilon_decay=0.999999,tau=0.05)
    agent_ddpg.load_model("models/bipedal_walker_ddpg_400.pth")
    for episode in range(episodes):
        state,info = env.reset()
        episode_reward = 0
        episode_loss = 0

        for t in range(max_timesteps):
            action = agent_ddpg.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            agent_ddpg.replay_buffer.add((state, action, reward, next_state,terminated or truncated))
            actor_loss = agent_ddpg.train()

            state = next_state
            episode_reward += reward
            episode_loss += actor_loss
            
            if terminated or truncated:
                break
        if episode % save_freq == 0:
            agent_ddpg.save_model(episode,f"models/bipedal_walker_ddpg_{episode}.pth")
        print(f"Episode {episode+1}/{episodes}, Reward: {episode_reward}, noise:{agent_ddpg.epsilon}")

        writer.add_scalar('reward', episode_reward, episode)
        writer.add_scalar('loss', episode_loss, episode)

def train_sac():
    env = gym.make("BipedalWalker-v3",hardcore=True, render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high = env.action_space.high[0]
    action_low = env.action_space.low[0]

    sac = SAC(state_dim, action_dim, action_high, action_low)
    replay_buffer = ReplayBuffer()

    episodes = 1000
    max_timesteps = 2000
    batch_size = 256
    for episode in range(episodes):
        state,info = env.reset()
        episode_reward = 0
        episode_loss = 0
        actor_loss = 0
        for t in range(max_timesteps):
            action = sac.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            replay_buffer.add((state, action, reward, next_state, terminated or truncated))

            if replay_buffer.size() > batch_size:
                actor_loss = sac.update(replay_buffer, batch_size)

            state = next_state
            episode_reward += reward
            episode_loss += actor_loss

            if terminated or truncated:
                break
        if episode % 100 == 0:
            sac.save_model(episode,f"models/bipedal_walker_sac_{episode}.pth")
        writer.add_scalar('reward', episode_reward, episode)
        writer.add_scalar('loss', episode_loss, episode)
        print(f'Episode {episode+1}/{episodes}, Reward: {episode_reward}')

if __name__=="__main__":
    # train()
    # test_env()
    train_sac()