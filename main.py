from environment.game_env import GameEnv
from agent.q_learning_agent import QLearningAgent

env = GameEnv()

state_size = 4
action_size = 3

agent = QLearningAgent(state_size, action_size)

episodes = 50

for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    while True:
        action = agent.choose_action(state)
        
        next_state, reward, done = env.step(action)
        
        agent.learn(state, action, reward, next_state)

        state = next_state
        total_reward += reward

        if done:
            break

    print(f"Episode {episode+1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")