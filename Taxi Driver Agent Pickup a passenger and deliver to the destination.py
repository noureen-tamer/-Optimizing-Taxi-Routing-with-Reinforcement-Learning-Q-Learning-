import gym
import numpy as np
import random
import pygame
import time

# ===============================
# 1. Q-Learning Agent
# ===============================
class TaxiQLearningAgent:
    def __init__(self, env, learning_rate=0.7, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize Q-table: states x actions
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update_q(self, state, action, reward, next_state, done):
        max_next = np.max(self.q_table[next_state])
        target = reward + self.gamma * max_next * (not done)
        self.q_table[state][action] += self.lr * (target - self.q_table[state][action])

    def train(self, episodes=5000):
        rewards = []
        for ep in range(episodes):
            state = self.env.reset()[0] ## 1st reset the enviroment 
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(state) ## 2nd choose the action given the current state
                next_state, reward, done, truncated, _ = self.env.step(action) ## 3rd enviroment respond 
                done = done or truncated

                self.update_q(state, action, reward, next_state, done) ## 4th update the q-table
                state = next_state
                total_reward += reward ## got from the enviroment 

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            rewards.append(total_reward) ## append all the rewards

            if (ep+1) % 500 == 0:
                avg = np.mean(rewards[-500:])
                print(f"Episode {ep+1}, Average Reward: {avg:.2f}, Epsilon: {self.epsilon:.3f}")
        print("Training completed!")

# ===============================
# 2. Simple Pygame GUI
# ===============================
class TaxiGUI:
    def __init__(self, env, cell_size=80):
        pygame.init()
        self.env = env
        self.cell_size = cell_size
        self.width = 5 * cell_size
        self.height = 5 * cell_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Taxi-v3 Q-learning GUI")
        self.colors = { ## colors are RGB
            "road": (200, 200, 200),
            "taxi": (255, 200, 0),
            "passenger": (0, 0, 255),
            "destination": (0, 255, 0)
        }
        # Fixed passenger/destination locations
        self.locs = [(0,0),(0,4),(4,0),(4,3)]

    def decode_state(self, state):
        dest_loc = state % 4
        state //= 4
        pass_loc = state % 5
        state //= 5
        taxi_col = state % 5
        taxi_row = state // 5
        return taxi_row, taxi_col, pass_loc, dest_loc

    def draw_grid(self, state):
        self.screen.fill((0,0,0))
        taxi_row, taxi_col, pass_loc, dest_loc = self.decode_state(state)

        # Draw cells
        for r in range(5):
            for c in range(5):
                pygame.draw.rect(self.screen, self.colors["road"],
                                 (c*self.cell_size, r*self.cell_size, self.cell_size, self.cell_size), 0)
                pygame.draw.rect(self.screen, (0,0,0),
                                 (c*self.cell_size, r*self.cell_size, self.cell_size, self.cell_size), 1)

        # Draw passenger if not in taxi
        if pass_loc != 4: ## where the 4 loc means that the passenger is inside the taxi so it isnot drawn
            pr, pc = self.locs[pass_loc]
            pygame.draw.circle(self.screen, self.colors["passenger"],
                               (pc*self.cell_size + self.cell_size//2, pr*self.cell_size + self.cell_size//2), self.cell_size//3)

        # Draw destination
        dr, dc = self.locs[dest_loc]
        pygame.draw.rect(self.screen, self.colors["destination"],
                         (dc*self.cell_size+10, dr*self.cell_size+10, self.cell_size-20, self.cell_size-20))

        # Draw taxi
        pygame.draw.rect(self.screen, self.colors["taxi"],
                         (taxi_col*self.cell_size+15, taxi_row*self.cell_size+15, self.cell_size-30, self.cell_size-30))
        pygame.display.update()

    def play(self, agent, episodes=5, delay=0.9): ## epsisode 5 as it will pickup and drive the passenger 5 times
        for ep in range(episodes):
            state = self.env.reset()[0]
            done = False
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                action = np.argmax(agent.q_table[state])
                state, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated
                self.draw_grid(state)
                time.sleep(delay)
            print(f"Episode {ep+1} completed")
        pygame.quit()


# ===============================
# 3. Run Everything
# ===============================
if __name__ == "__main__":
    env = gym.make("Taxi-v3")
    agent = TaxiQLearningAgent(env)
    print("Training Q-learning agent...")
    agent.train(episodes=5000)

    print("Launching GUI demo...")
    gui = TaxiGUI(env)
    gui.play(agent, episodes=5, delay=0.5)