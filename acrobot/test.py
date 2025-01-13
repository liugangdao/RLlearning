import gymnasium as gym
from agent import DQNNetwork
import torch
import imageio

def render_test(device,path,gif_path):
    device_ = torch.device(device)
    env = gym.make('Acrobot-v1',render_mode="rgb_array")
    max_step = 500
    state,info = env.reset()
    is_ok,final_step = False,0

    frames = []

    model = DQNNetwork(6,3)
    model.load_state_dict(torch.load(path,weights_only=True))
    model = model.to(device_)
    for _ in range(max_step):
        final_step += 1
        # env.render()
        img = env.render()  # 获取图像
        frames.append(img) 
        
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device_)
        with torch.no_grad():
            q_values = model(state)
            # print(model_output)
            # 得到action
            action_probs = torch.softmax(q_values, dim=-1)
            action = torch.argmax(action_probs).item()
            

        state,reward,terminated , truncated, info = env.step(action)
        if terminated:
            is_ok = True
            break
    
    env.close()

    with imageio.get_writer(gif_path, mode='I', duration=0.05) as writer:
        for frame in frames:
            writer.append_data(frame)  # 将每一帧图像写入 GIF 文件
    return is_ok,final_step

if __name__=="__main__":
    device = "cuda:1"
    path = "acrobot_dqn_8400.pth"
    
    print(render_test(device,path,"acrobot_test.gif"))

