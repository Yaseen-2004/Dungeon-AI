import random

class GameEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.player_health = 100
        self.player_level = 1
        self.score = 0
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        health_status = 0
        
        if self.player_health > 70:
            health_status = 2   # healthy
        elif self.player_health > 30:
            health_status = 1   # medium
        else:
            health_status = 0   # critical

        return [
            health_status,
            self.player_level,
            self.steps
        ]

    def step(self, action):
        """
        Actions:
        0 → Easy Enemy
        1 → Hard Enemy
        2 → Heal Player
        """

        reward = 0
        done = False

        if action == 0:  # easy enemy
            damage = random.randint(5, 10)
            self.player_health -= damage
            
            if self.player_health > 30:
                reward = 5
            else:
                reward = -5

        elif action == 1:  # hard enemy
            damage = random.randint(10, 20)
            self.player_health -= damage
            
            if self.player_health > 50:
                reward = 10
            else:
                reward = -10

        elif action == 2:  # heal
            heal = random.randint(5, 15)
            self.player_health += heal
            
            if self.player_health < 50:
                reward = 8
            else:
                reward = -2

        # Clamp health
        self.player_health = min(100, max(0, self.player_health))

        # Update score
        self.score += reward
        self.steps += 1

        # Game over condition
        if self.player_health <= 0:
            done = True
            reward = -20

        return self._get_state(), reward, done