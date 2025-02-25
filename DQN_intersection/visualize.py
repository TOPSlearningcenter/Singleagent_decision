import matplotlib.pyplot as plt
import os

# 简单的移动平均函数
def moving_average(data, window_size):
    return [sum(data[i:i+window_size]) / window_size for i in range(len(data) - window_size + 1)]

def visualize_results():
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

    episode_rewards = []  # 用于存储每个回合的总奖励
    episode_epsilon = []  # 用于存储每个回合的epsilon值

    # 从训练日志文件中读取奖励和epsilon值
    with open(os.path.join(model_dir, 'training_log.txt'), 'r') as log_file:
        for line in log_file:
            reward, epsilon = line.strip().split(", ")  # 分割每一行的奖励和epsilon
            episode_rewards.append(float(reward.split(": ")[1]))  # 存储奖励值
            episode_epsilon.append(float(epsilon.split(": ")[1]))  # 存储epsilon值

    # 对奖励进行平滑处理
    smoothed_rewards = moving_average(episode_rewards, window_size=10)  # 使用窗口大小为10的移动平均

    # 创建一个图形窗口
    plt.figure(figsize=(12, 6))

    # 绘制第一个子图：每个回合的奖励（平滑后的）
    plt.subplot(1, 2, 1)
    plt.plot(smoothed_rewards, label='Smoothed Reward')
    plt.title('Smoothed Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Total Reward')

    # 绘制第二个子图：epsilon的衰减
    plt.subplot(1, 2, 2)
    plt.plot(episode_epsilon, label='Epsilon', color='r')
    plt.title('Epsilon Decay per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')

    # 调整图像的布局，使得图形不会重叠
    plt.tight_layout()

    # 显示图形
    plt.show()
