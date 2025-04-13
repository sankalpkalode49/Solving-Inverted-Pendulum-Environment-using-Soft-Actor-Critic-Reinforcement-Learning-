import pybullet_envs
import gym
import numpy as np
import matplotlib.pyplot as plt  # Added here
from sac_torch import agent
import os

if __name__ == "__main__":
    # Create environment and agent
    env = gym.make('InvertedPendulumBulletEnv-v0')
    agent = agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0])
    env.render(mode='human')
    n_games = 500
    filename = 'inverted_pendulum.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    # Load pre-trained model if necessary
    if load_checkpoint:
        agent.load_models()
        # env.render(mode='human')

    # Start training
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_

        # Track score history and moving average
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        # Render every 10th episode
        if i % 10 == 0:  # You can adjust this value to control how often you render
            print(f"Rendering at episode {i}")
            env.render(mode='human')  # Render the environment every 10 episodes

        print(f'Episode: {i}, Score: {score}, Avg Score: {avg_score}')

    # Plot directly using matplotlib
    x = [i + 1 for i in range(n_games)]
    running_avg = np.zeros(len(score_history))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(score_history[max(0, i - 100):(i + 1)])

    # Ensure the plots directory exists
    os.makedirs('plots', exist_ok=True)

    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(x, score_history, label='Score per Episode', color='orange', alpha=0.4)
    plt.plot(x, running_avg, label='Running Average (100 episodes)', color='blue', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('SAC on InvertedPendulumBulletEnv')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(figure_file)
    plt.close()

