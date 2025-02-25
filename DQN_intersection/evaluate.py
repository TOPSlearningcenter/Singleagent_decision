# evaluate.py
import gymnasium as gym
import highway_env
import os
import torch
import torch.nn as nn

def evaluate_model():
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

    env = gym.make("intersection-v0", render_mode='human')
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n

    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
    model = DQN(state_size, action_size)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'dqn_model.pth')))
    model.eval()

    def choose_action(state):
        if isinstance(state, tuple):
            state = state[0]
        state = torch.FloatTensor(state).unsqueeze(0)
        state = state.view(state.size(0), -1)
        q_values = model(state)
        return torch.argmax(q_values).item()

    def evaluate():
        total_reward = 0
        state = env.reset()
        done = False
        while not done:
            action = choose_action(state)
            next_state, reward, done, _, info = env.step(action)
            state = next_state
            total_reward += reward

        print(f"Total reward in evaluation: {total_reward}")

    evaluate()
