import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class G03Agent:
    def __init__(self, env, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, learning_rate=1e-3, buffer_size=10000):
        """
        Initialize the agent for the given environment.

        Args:
            env: The OurHexGame environment instance.
        """
        # Extract dimensions from the environment
        self.state_size = env.observation_spaces["player_1"]["observation"].shape[0] * env.observation_spaces["player_1"]["observation"].shape[1]
        self.action_size = env.action_spaces["player_1"].n

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-network
        self.q_network = QNetwork(self.state_size, self.action_size).to(self.device)
        self.target_network = QNetwork(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # Replay buffer
        self.memory = deque(maxlen=buffer_size)

        # Synchronize target network
        self.update_target_network()

    def update_target_network(self):
        """Copy weights from the main Q-network to the target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, observation, reward, termination, truncation, info):
        """
        Select an action based on the current observation.
        Args:
            observation: The observation dictionary from env.last().
            reward: Reward signal (ignored in this context).
            termination: Termination flag for the current agent.
            truncation: Truncation flag for the current agent.
            info: Additional information, including the action_mask.
        Returns:
            int: Selected action.
        """
        # Flatten the observation board
        state = observation["observation"].flatten()
        action_mask = info["action_mask"]  # Access action_mask from info

        if random.random() < self.epsilon:
            # Choose a random valid action
            valid_actions = np.where(action_mask == 1)[0]
            return int(random.choice(valid_actions))

        # Predict Q-values for all actions
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy()[0]

        # Mask invalid actions by setting their Q-values to a very low value
        q_values = np.where(action_mask, q_values, -np.inf)
        return int(np.argmax(q_values))

    def train(self, batch_size=64):
        """Train the Q-network using experiences from the replay buffer."""
        if len(self.memory) < batch_size:
            return

        # Sample a minibatch of experiences
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        # Convert data to tensors
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        # Compute current Q-values and target Q-values
        q_values = self.q_network(state).gather(1, action).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(next_state).max(1)[0]
            targets = reward + (1 - done) * self.gamma * next_q_values

        # Compute loss and update the Q-network
        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon for exploration
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save_model(self, file_name="gxx_agent.pth"):
        """Save the Q-network model."""
        torch.save(self.q_network.state_dict(), file_name)

    def load_model(self, file_name="gxx_agent.pth"):
        """Load the Q-network model."""
        self.q_network.load_state_dict(torch.load(file_name))
        self.update_target_network()
