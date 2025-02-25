# train.py
import gymnasium as gym
import highway_env
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os

def train_model():
    # DQN 网络模型定义
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

    # 环境和超参数
    env = gym.make("intersection-v0", render_mode='human')
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n
    batch_size = 32
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    learning_rate = 0.001
    target_model_update_freq = 10
    memory = deque(maxlen=2000)

    model = DQN(state_size, action_size)
    target_model = DQN(state_size, action_size)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def choose_action(state):
        if np.random.rand() <= epsilon:
            return random.randrange(action_size)
        if isinstance(state, tuple):
            state = state[0]
        state = torch.FloatTensor(state).unsqueeze(0)
        state = state.view(state.size(0), -1)
        q_values = model(state)
        return torch.argmax(q_values).item()

    def experience_replay():
        if len(memory) < batch_size:
            return
        batch = random.sample(memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states).view(batch_size, -1)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states).view(batch_size, -1)
        dones = torch.BoolTensor(dones)

        current_q_values = model(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q_values = target_model(next_states).max(1)[0]
        target_q_values = rewards + (gamma * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values, target_q_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    episode_rewards = []
    episode_epsilon = []
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
    os.makedirs(model_dir, exist_ok=True)

    episodes = 2000 #2000
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = choose_action(state)
            next_state, reward, done, _, info = env.step(action)

            if isinstance(state, tuple):
                state = state[0]

            memory.append((state, action, reward, next_state, done))
            experience_replay()

            state = next_state
            total_reward += reward

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if episode % target_model_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        episode_rewards.append(total_reward)
        episode_epsilon.append(epsilon)

        print(f"Episode {episode}/{episodes} - Total reward: {total_reward}, Epsilon: {epsilon}")

    torch.save(model.state_dict(), os.path.join(model_dir, 'dqn_model.pth'))
    with open(os.path.join(model_dir, 'training_log.txt'), 'w') as log_file:
        for reward, eps in zip(episode_rewards, episode_epsilon):
            log_file.write(f"Reward: {reward}, Epsilon: {eps}\n")

    print("Model and logs saved successfully!")

