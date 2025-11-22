🚕 Taxi-v3 Route Optimization with Q-Learning

A reinforcement learning project that trains an agent to pick up and drop off passengers efficiently in the OpenAI Gym Taxi-v3 environment.
The agent learns using Q-Learning, a model-free algorithm that updates a Q-table based on environment feedback.

This project includes:

✔ Q-learning training over 2,000 episodes
✔ Policy extraction
✔ Testing the learned policy
✔ Episode reward logging
✔ Returning episode_total_reward ≥ 4 in the evaluation phase
✔ Frames captured from env.render()

📌 Project Overview

The goal is to teach an agent how to navigate a 5×5 grid:

Pick up a passenger from one of four fixed locations

Drop them off at the designated location

Minimize the number of steps and avoid illegal moves


The agent learns by:

Exploring random actions early (high ε)

Exploiting learned actions later (low ε)

Updating Q-values after every step

🧠 Reinforcement Learning Algorithm: Q-Learning

The Q-Learning update rule used:

Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]

Where:

s = current state

a = action taken

r = reward

s' = next state

α = learning rate

γ = discount factor
🛠 Requirements

Install dependencies:

pip install gym pygame numpy

Note:
Gym versions ≥0.26 use the new API (reset()[0], step() returns terminated & truncated).

📊 Training Performance

Expected behavior:

First 500 episodes: negative rewards while exploring

After ~1000 episodes: positive rewards

Final mean rewards: 6–9

Epsilon decays to 0.01

📘 Understanding the Environment

States (500 total)

Each state is encoded as a single number representing:

Taxi row (0–4)

Taxi column (0–4)

Passenger location (0–4, where 4 = inside taxi)

Destination location (0–3)
Actions (6 total)
0 → Move south
1 → Move north
2 → Move east
3 → Move west
4 → Pick up
5 → Drop off
📥 Example Output

Training finished!
Episode total reward from test episode: 5
