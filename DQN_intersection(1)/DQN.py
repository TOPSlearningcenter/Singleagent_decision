import gymnasium as gym
import highway_env
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt


# 初始化环境
env = gym.make("intersection-v0", render_mode='human')

# # 加载整个模型
# model = torch.load('dqn_model.pth')
# model.eval()  # 设置为评估模式

# DQN 网络模型
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 超参数
state_size = env.observation_space.shape[0] * env.observation_space.shape[1] # 状态空间维度
action_size = env.action_space.n  # 动作空间维度
batch_size = 32
gamma = 0.99  # 折扣因子
epsilon = 1.0  # 探索率
epsilon_min = 0.01  # 最小探索率
epsilon_decay = 0.995  # 探索率衰减
learning_rate = 0.001  # 学习率
target_model_update_freq = 10  # 目标网络更新频率
memory = deque(maxlen=2000)  # 经验回放池

# 创建DQN模型和目标网络
model = DQN(state_size, action_size)
target_model = DQN(state_size, action_size)
target_model.load_state_dict(model.state_dict())  # 初始化目标网络与训练网络相同
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 选择动作
def choose_action(state):
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)  # 随机选择动作，探索
    if isinstance(state, tuple):
        state = state[0]  # 假设状态是一个元组，提取第一个部分
    state = torch.FloatTensor(state).unsqueeze(0)
    state = state.view(state.size(0), -1)
    q_values = model(state)  # 计算每个动作的Q值
    return torch.argmax(q_values).item()  # 返回Q值最大的动作


# 经验回放
def experience_replay():
    if len(memory) < batch_size:
        return

    # 随机采样一个batch
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.BoolTensor(dones)

    states = states.view(states.size(0), -1)  # 展平为 (32, 15 * 7)
    next_states = next_states.view(next_states.size(0), -1)  # 展平为 (32, 15 * 7)

    # 确保 actions 是一维的，且模型输出的 Q 值形状为 (batch_size, num_actions)
    actions = actions.unsqueeze(1)  # 将 actions 的形状从 (batch_size,) 转为 (batch_size, 1)

    # 使用 gather() 来获取每个状态下选定动作的 Q 值
    current_q_values = model(states).gather(1, actions).squeeze(1)  # 去掉多余的维度

    # 下一状态的最大Q值
    with torch.no_grad():
        next_q_values = target_model(next_states).max(1)[0]

    # 计算目标Q值
    target_q_values = rewards + (gamma * next_q_values * ~dones)

    # 计算损失
    loss = nn.MSELoss()(current_q_values, target_q_values)

    # 更新模型
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 用于存储每个episode的reward和epsilon
episode_rewards = []
episode_epsilon = []
# 训练循环
episodes = 500
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = choose_action(state)
        next_state, reward, done, _, info= env.step(action)

        if isinstance(state, tuple):
            state = state[0]

        # 存储经验
        memory.append((state, action, reward, next_state, done))

        # 训练DQN
        experience_replay()

        state = next_state
        total_reward += reward

        # 存储每个episode的reward和epsilon
        episode_rewards.append(total_reward)
        episode_epsilon.append(epsilon)

    # 更新epsilon以减少探索
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # 每隔一定步数更新目标网络
    if episode % target_model_update_freq == 0:
        target_model.load_state_dict(model.state_dict())

    print(f"Episode {episode}/{episodes} - Total reward: {total_reward}, Epsilon: {epsilon}")

torch.save(model.state_dict(), 'dqn_model.pth')
print("Model saved successfully!")

# 绘制图形
plt.figure(figsize=(12, 6))

# 绘制reward曲线
plt.subplot(1, 2, 1)
plt.plot(episode_rewards, label='Reward')
plt.title('Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

# 绘制epsilon曲线
plt.subplot(1, 2, 2)
plt.plot(episode_epsilon, label='Epsilon', color='r')
plt.title('Epsilon Decay per Episode')
plt.xlabel('Episode')
plt.ylabel('Epsilon')

plt.tight_layout()
plt.show()