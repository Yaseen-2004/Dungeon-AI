from environment.game_env import GameEnv
from agent.q_learning_agent import QLearningAgent
import time

env = GameEnv()

state_size = 3
action_size = 3

agent = QLearningAgent(state_size, action_size)

# 🔥 Train quickly before playing
print("Training AI Dungeon Master...\n")

for episode in range(200):
    state = env.reset()
    
    while True:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state

        if done:
            break

# Stop exploration → pure intelligence
agent.epsilon = 0

print("Training Complete! Starting Game...\n")
time.sleep(1)

# 🎮 PLAY MODE
state = env.reset()

while True:
    print("\n==========================")
    print(f"Player Health: {env.player_health}")
    print("==========================")

    # AI decision
    action = agent.choose_action(state)

    if action == 0:
        print("🤖 AI spawned: EASY ENEMY")
    elif action == 1:
        print("🤖 AI spawned: HARD ENEMY")
    elif action == 2:
        print("🤖 AI gave you: HEAL")

    input("Press Enter to continue...")

    next_state, reward, done = env.step(action)

    state = next_state

    if done:
        print("\n💀 You Died! Game Over!")
        break