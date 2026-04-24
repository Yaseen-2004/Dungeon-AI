import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Q-table (state → action values)
        self.q_table = {}

        # Hyperparameters
        self.alpha = 0.1          # learning rate
        self.gamma = 0.9          # discount factor
        self.epsilon = 1.0        # exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    # ----------------------------
    # Convert state → hashable key
    # ----------------------------
    def _get_state_key(self, state):
        return tuple(state)

    # ----------------------------
    # Choose action (ε-greedy)
    # ----------------------------
    def choose_action(self, state):
        state_key = self._get_state_key(state)

        # Initialize state in Q-table if not present
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)

        # Exploration
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        # Exploitation
        return np.argmax(self.q_table[state_key])

    # ----------------------------
    # Learning step (Q-update)
    # ----------------------------
    def learn(self, state, action, reward, next_state):
        state_key = self._get_state_key(state)
        next_key = self._get_state_key(next_state)

        # Ensure states exist in Q-table
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)

        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.action_size)

        # Best next action value
        best_next_action = np.max(self.q_table[next_key])

        # Q-learning update formula
        self.q_table[state_key][action] += self.alpha * (
            reward + self.gamma * best_next_action
            - self.q_table[state_key][action]
        )

        # ----------------------------
        # Decay exploration rate
        # ----------------------------
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # ----------------------------
    # Utility: get policy (for debug)
    # ----------------------------
    def get_policy(self, state):
        state_key = self._get_state_key(state)

        if state_key not in self.q_table:
            return None

        return np.argmax(self.q_table[state_key])

    # ----------------------------
    # Utility: reset exploration
    # ----------------------------
    def reset_exploration(self):
        self.epsilon = 1.0