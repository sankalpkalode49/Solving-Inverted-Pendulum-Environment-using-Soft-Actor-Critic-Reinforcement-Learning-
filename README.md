🚀 **Soft Actor-Critic (SAC) on Inverted Pendulum using PyBullet**  
This project implements the Soft Actor-Critic (SAC) algorithm to train an agent to balance an inverted pendulum using the `InvertedPendulumBulletEnv-v0` environment from `pybullet_envs`. The implementation is done from scratch using PyTorch, including custom neural network architectures, replay buffer, and SAC training logic.

---

📁 **Project Structure**


---

🧠 **Algorithm Overview**  
Soft Actor-Critic (SAC) is an off-policy actor-critic algorithm that optimizes a stochastic policy in an entropy-regularized reinforcement learning framework. The entropy term encourages exploration by discouraging premature convergence to suboptimal deterministic policies.

---

✨ **Key Features**

- ✅ Twin Q-networks to reduce overestimation bias  
- ✅ Value network for stable training  
- ✅ Stochastic actor with entropy tuning  
- ✅ Experience replay for sample efficiency  

---

📦 **Requirements**  
Make sure you have the following installed:

```bash
pip install torch gym numpy matplotlib pybullet
sudo apt-get install xvfb


🏋️ How to Train
Train the SAC agent on the inverted pendulum:
python3 main_sac.py '''

During training, model checkpoints and a plot of the reward curve will be saved.

🎮 How to Run Inference
To visualize the trained agent:

python3 inference.py

This will load the saved model and render the environment so you can see the pendulum balancing in real time.
