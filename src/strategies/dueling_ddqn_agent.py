"""
Dueling Double Deep Q-Network (Dueling DDQN) Agent (With Action Masking)
------------------------------------------------------------------------
PyTorch-native Dueling DDQN with Action Masking support, Experience Replay,
Polyak Soft Sync, and Epsilon-Greedy exploration tailored for quantitative trading.
"""
from __future__ import annotations
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network Architecture:
    Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # State Value Stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Advantage Stream A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_net(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        # Combine using mean-centered advantage formula
        q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values


class ReplayBuffer:
    """Fast deque-based experience replay buffer."""

    def __init__(self, capacity: int = 150_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.tensor(np.array(state), dtype=torch.float32, device=DEVICE),
            torch.tensor(action, dtype=torch.long, device=DEVICE).unsqueeze(1),
            torch.tensor(reward, dtype=torch.float32, device=DEVICE).unsqueeze(1),
            torch.tensor(np.array(next_state), dtype=torch.float32, device=DEVICE),
            torch.tensor(done, dtype=torch.float32, device=DEVICE).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


class DuelingDDQNAgent:
    """
    Dueling Double DQN Agent with Action Masking and Double Q-learning updates.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9995,
        buffer_size: int = 150_000,
        batch_size: int = 128,
        tau: float = 0.005,  # Polyak soft update rate
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Networks
        self.q_net = DuelingQNetwork(state_dim, action_dim, hidden_dim).to(DEVICE)
        self.target_net = DuelingQNetwork(state_dim, action_dim, hidden_dim).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=lr, weight_decay=1e-4)
        self.loss_fn = nn.SmoothL1Loss()  # Huber Loss

        self.memory = ReplayBuffer(capacity=buffer_size)
        self.step_count = 0

    def select_action(self, state: np.ndarray, action_mask: np.ndarray | None = None, evaluate: bool = False) -> int:
        """
        Select action using epsilon-greedy policy with Action Masking support.
        """
        if action_mask is not None:
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) == 0:
                valid_actions = [0]
        else:
            valid_actions = list(range(self.action_dim))

        if not evaluate and random.random() < self.epsilon:
            return int(random.choice(valid_actions))

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            q_values = self.q_net(state_t)[0]

            if action_mask is not None:
                # Mask out invalid actions by setting Q-values to -infinity
                mask_t = torch.tensor(action_mask, dtype=torch.bool, device=DEVICE)
                q_values[~mask_t] = -1e9

            return int(q_values.argmax(dim=-1).item())

    def update(self) -> float | None:
        """Perform one training update step."""
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # 1. Current Q values for chosen actions
        curr_q = self.q_net(states).gather(1, actions)

        # 2. Double DQN target: Online Network selects action, Target Network evaluates
        with torch.no_grad():
            best_next_actions = self.q_net(next_states).argmax(dim=-1, keepdim=True)
            next_q_targets = self.target_net(next_states).gather(1, best_next_actions)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q_targets

        # 3. Compute loss & optimize
        loss = self.loss_fn(curr_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # 4. Soft update target network (Polyak)
        for target_param, q_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_param.data)

        # 5. Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_end)

        self.step_count += 1
        return float(loss.item())

    def save(self, filepath: str):
        """Save model weights."""
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "epsilon": self.epsilon,
                "step_count": self.step_count,
            },
            filepath,
        )

    def load(self, filepath: str):
        """Load model weights."""
        checkpoint = torch.load(filepath, map_location=DEVICE)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self.step_count = checkpoint.get("step_count", 0)
