import gym
import torch
from sac_torch import agent
import pybullet_envs
# Create the environment
try:
    env = gym.make('InvertedPendulumBulletEnv-v0')
    print("Environment initialized successfully")
    env.render(mode='human')
except Exception as e:
    print(f"Failed to create environment: {e}")
    exit(1)

# Initialize the agent
sac_agent = agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0])

# Load the trained models
try:
    sac_agent.load_models()
    print("Models loaded successfully")
except Exception as e:
    print(f"Failed to load models: {e}")
    exit(1)

# Reset the environment
state = env.reset()
print("Environment reset, starting inference loop")

# Run inference
done = False
step_count = 0
while not done:
    action = sac_agent.choose_action(state)
    print(f"Step {step_count}: Action = {action}")  # Debug action output
    state, reward, done, info = env.step(action)
    try:
        env.render()
        print(f"Step {step_count}: Reward = {reward}, Done = {done}")
    except Exception as e:
        print(f"Rendering failed at step {step_count}: {e}")
        break
    step_count += 1

print(f"Inference completed after {step_count} steps")
env.close()