import pygame
import sys
import time
import matplotlib.pyplot as plt

from environment.game_env import GameEnv
from agent.q_learning_agent import QLearningAgent

# ---------------- INIT ----------------
pygame.init()

WIDTH, HEIGHT = 800, 500
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("🧠 AI Dungeon Master - RL Demo")

clock = pygame.time.Clock()

# ---------------- COLORS ----------------
BG = (18, 18, 30)
WHITE = (240, 240, 240)
RED = (220, 60, 60)
GREEN = (60, 220, 120)
BLUE = (80, 140, 255)
GRAY = (130, 130, 130)
YELLOW = (255, 210, 80)

# ---------------- FONTS ----------------
title_font = pygame.font.SysFont("arial", 40, bold=True)
font = pygame.font.SysFont("arial", 24)
small_font = pygame.font.SysFont("arial", 18)

# ---------------- GAME STATES ----------------
MENU = "menu"
PLAY = "play"
GAME_OVER = "game_over"

game_state = MENU

# ---------------- AI + ENV ----------------
env = GameEnv()
agent = QLearningAgent(state_size=3, action_size=3)

# ---------------- TRAIN FUNCTION ----------------
def train_agent():
    for _ in range(150):
        state = env.reset()
        while True:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            if done:
                break
    agent.epsilon = 0

train_agent()

state = env.reset()

# ---------------- TRACKING ----------------
rewards_history = []
episode_reward = 0
graph_surface = None
last_action = ""

# ---------------- GRAPH ----------------
def create_graph():
    global graph_surface

    if len(rewards_history) < 2:
        return

    plt.figure(figsize=(4, 2))
    plt.plot(rewards_history, color='blue')
    plt.title("AI Learning Progress")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.tight_layout()

    plt.savefig("graph.png")
    plt.close()

    try:
        graph_surface = pygame.image.load("graph.png")
        graph_surface = pygame.transform.scale(graph_surface, (320, 160))
    except:
        graph_surface = None

# ---------------- UI ----------------
def draw_button(text, x, y, w, h, color):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()

    rect = pygame.Rect(x, y, w, h)

    pygame.draw.rect(screen, color, rect, border_radius=10)

    label = font.render(text, True, WHITE)
    screen.blit(label, (x + 20, y + 10))

    if rect.collidepoint(mouse) and click[0]:
        time.sleep(0.2)  # debounce click
        return True
    return False


def draw_health_bar(x, y, w, h, value):
    value = max(0, min(100, value))

    pygame.draw.rect(screen, RED, (x, y, w, h), border_radius=8)
    pygame.draw.rect(screen, GREEN, (x, y, w * (value / 100), h), border_radius=8)
    pygame.draw.rect(screen, WHITE, (x, y, w, h), 2, border_radius=8)


def draw_panel():
    pygame.draw.rect(screen, (30, 30, 50), (50, 120, 700, 300), border_radius=12)
    pygame.draw.rect(screen, BLUE, (50, 120, 700, 300), 2, border_radius=12)

# ---------------- MAIN LOOP ----------------
running = True

while running:
    screen.fill(BG)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # ================= MENU =================
    if game_state == MENU:

        screen.blit(title_font.render("🧠 AI DUNGEON MASTER", True, WHITE), (200, 120))
        screen.blit(font.render("Reinforcement Learning Game Demo", True, GRAY), (220, 180))

        if draw_button("▶ START GAME", 300, 260, 200, 50, BLUE):
            env = GameEnv()
            state = env.reset()
            episode_reward = 0
            graph_surface = None
            game_state = PLAY

    # ================= PLAY =================
    elif game_state == PLAY:

        pygame.time.delay(500)

        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)

        episode_reward += reward
        state = next_state

        if action == 0:
            last_action = "⚔️ EASY ENEMY"
        elif action == 1:
            last_action = "🔥 HARD ENEMY"
        else:
            last_action = "💖 HEAL"

        screen.blit(title_font.render("AI DUNGEON RUN", True, WHITE), (230, 20))

        draw_panel()
        draw_health_bar(200, 180, 400, 30, env.player_health)

        screen.blit(font.render(f"Health: {env.player_health}", True, WHITE), (200, 220))
        screen.blit(font.render(last_action, True, YELLOW), (200, 270))
        screen.blit(small_font.render(f"Steps: {env.steps} | Score: {env.score}", True, GRAY), (200, 320))

        if done:
            rewards_history.append(episode_reward)
            episode_reward = 0
            game_state = GAME_OVER

    # ================= GAME OVER =================
    elif game_state == GAME_OVER:

        screen.blit(title_font.render("💀 GAME OVER", True, RED), (280, 80))
        screen.blit(font.render(f"Final Score: {env.score}", True, WHITE), (300, 150))

        create_graph()

        if graph_surface:
            screen.blit(graph_surface, (240, 220))

        if draw_button("🔁 RESTART", 300, 400, 200, 50, GREEN):
            train_agent()
            env = GameEnv()
            state = env.reset()
            episode_reward = 0
            game_state = PLAY

        if draw_button("🏠 MENU", 300, 460, 200, 50, BLUE):
            game_state = MENU

    pygame.display.update()
    clock.tick(60)

pygame.quit()
sys.exit()