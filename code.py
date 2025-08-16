# Install dependencies if needed
# pip install numpy gym torch matplotlib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# -----------------------------
# 1. Environment Simulation
# -----------------------------
class RetailEnv:
    def __init__(self, max_inventory=100, base_price=10):
        self.max_inventory = max_inventory
        self.base_price = base_price
        self.reset()

    def reset(self):
        self.inventory = self.max_inventory
        self.price = self.base_price
        self.total_profit = 0
        return np.array([self.inventory, self.price], dtype=np.float32)

    def step(self, action):
        """
        Action: 0 -> decrease price, 1 -> keep price, 2 -> increase price
        """
        if action == 0:
            self.price *= 0.9
        elif action == 2:
            self.price *= 1.1
        # Simulate demand: higher price -> lower demand
        demand = max(0, int((100 - self.price) + np.random.normal(0,5)))
        sold = min(demand, self.inventory)
        revenue = sold * self.price
        self.inventory -= sold
        self.total_profit += revenue
        done = self.inventory == 0
        reward = revenue  # reward = profit for the step
        state = np.array([self.inventory, self.price], dtype=np.float32)
        return state, reward, done

# -----------------------------
# 2. Deep Q-Network
# -----------------------------
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# -----------------------------
# 3. Training the Agent
# -----------------------------
env = RetailEnv()
state_size = 2
action_size = 3
dqn = DQN(state_size, action_size)
optimizer = optim.Adam(dqn.parameters(), lr=0.001)
criterion = nn.MSELoss()

memory = deque(maxlen=10000)
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 200

for e in range(episodes):
    state = env.reset()
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        if random.random() < epsilon:
            action = random.randint(0, action_size-1)
        else:
            action = torch.argmax(dqn(state_tensor)).item()
        next_state, reward, done = env.step(action)
        memory.append((state, action, reward, next_state, done))
        state = next_state

        # Sample random batch from memory for training
        if len(memory) >= 32:
            batch = random.sample(memory, 32)
            states_b, actions_b, rewards_b, next_states_b, dones_b = zip(*batch)
            states_b = torch.FloatTensor(states_b)
            actions_b = torch.LongTensor(actions_b).unsqueeze(1)
            rewards_b = torch.FloatTensor(rewards_b).unsqueeze(1)
            next_states_b = torch.FloatTensor(next_states_b)
            dones_b = torch.FloatTensor(dones_b).unsqueeze(1)

            q_values = dqn(states_b).gather(1, actions_b)
            q_next = dqn(next_states_b).max(1)[0].unsqueeze(1)
            q_target = rewards_b + gamma * q_next * (1 - dones_b)

            loss = criterion(q_values, q_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    if e % 20 == 0:
        print(f"Episode {e}, Total Profit: {env.total_profit:.2f}, Epsilon: {epsilon:.2f}")

# -----------------------------
# 4. Evaluation
# -----------------------------
# Run a final simulation to see profit and inventory usage
state = env.reset()
done = False
profits = []
while not done:
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    action = torch.argmax(dqn(state_tensor)).item()
    state, reward, done = env.step(action)
    profits.append(reward)

total_profit = sum(profits)
overstock_remaining = env.inventory
print(f"Total Profit: {total_profit:.2f}, Overstock Remaining: {overstock_remaining}")
