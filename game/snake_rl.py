import tkinter as tk
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from collections import deque
import matplotlib.pyplot as plt

# === CONFIG ===
BOX_SIZE = 30
GRID_WIDTH = 17 # default 17
GRID_HEIGHT = 15 # default 15
TRAINING_MODE = False  # True = no visuals, just AI training
LOAD_MODEL = True  # Load an existing trained model
MODEL_PATH = "models/pptx/Beispiel 5/model.pth"  # Name of the model to load or save
SPEED = 1 if TRAINING_MODE else 50

EPISODES = 15000 # Number of episodes for training

# === REWARDS ===
MOVE_LOITER_PENALTY = -1
COLLISION_PENALTY = -100
APPLE_REWARD = 100
LOOP_PENALTY = -1  # Penalty for repetitive movement patterns
LOOP_WINDOW = 15   # Number of steps to look back for loops
MIN_LOOP_LENGTH = 3  # Minimum length of action sequence to consider for loops

# Exploration parameters
EPSILON_DECAY = 0.99945  # Decay rate for epsilon, default is 0.995
MIN_EPSILON = 0.01  # Minimum value for epsilon
TARGET_UPDATE_FREQ = 50  # Frequency of target network updates

# === METRICS STORAGE ===
episode_scores = []
episode_rewards = []
training_log = []

# === PLOTTING ===
PLOTTINGFREQ = 20  # Frequency of plotting training metrics

# === DEVICE CONFIGURATION ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


pending_loop = None

# === Q-NETWORK ===
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.model(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_network = QNetwork(state_size, action_size).to(device)
        self.target_network = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=2000)
        self.batch_size = 64
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = MIN_EPSILON
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

    def act(self, state):
        if TRAINING_MODE and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            return torch.argmax(self.q_network(state_tensor)).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        current_q = self.q_network(states).gather(1, actions)
        next_q = self.target_network(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, path=MODEL_PATH):
        torch.save(self.q_network.state_dict(), path)

    def load_model(self, path=MODEL_PATH):
        if os.path.exists(path):
            self.q_network.load_state_dict(torch.load(path, map_location=device))
            self.q_network.eval()
            print(f"Loaded model from {path}")

# === INIT ===
agent = DQNAgent(state_size=8, action_size=3)
if LOAD_MODEL:
    agent.load_model()

# === GAME STATE ===
score = 0
highscore = 0
snake_length = 3
snake_body = []
direction = "d"
game_running = True
move_reward = 0
episode_reward = 0
apple_pos = (0, 0)
action_history = deque(maxlen=LOOP_WINDOW)
position_history = deque(maxlen=LOOP_WINDOW)

# === TK INTERFACE === (Only when not in training mode)
if not TRAINING_MODE:
    root = tk.Tk()
    root.title("Snake")
    canvas = tk.Canvas(root, width=BOX_SIZE * GRID_WIDTH, height=BOX_SIZE * GRID_HEIGHT)
    canvas.pack()
    score_label = tk.Label(root, text=f"Score: {score}", font=("Arial", 16))
    score_label.pack()
    highscore_label = tk.Label(root, text=f"Highscore: {highscore}", font=("Arial", 16))
    highscore_label.pack()
    data_label = tk.Label(root, text="", font=("Arial", 12))
    data_label.pack()
    restart_button = tk.Button(root, text="Restart", font=("Arial", 14), command=lambda: restart_game())
    restart_button.pack_forget()

    # Grid and Apple setup for GUI
    grid_rects = []
    for row in range(GRID_HEIGHT):
        row_rects = []
        for col in range(GRID_WIDTH):
            x1, y1 = col * BOX_SIZE, row * BOX_SIZE
            x2, y2 = x1 + BOX_SIZE, y1 + BOX_SIZE
            color = "white" if (row + col) % 2 == 0 else "#d3ffcc"
            rect = canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
            row_rects.append(rect)
        grid_rects.append(row_rects)
    apple_rect = canvas.create_oval(0, 0, BOX_SIZE, BOX_SIZE, fill="red", outline="")
else:
    # Initialize snake and apple positions for training mode
    snake_body = [(GRID_WIDTH // 2 - i - 4, GRID_HEIGHT // 2) for i in range(snake_length)]
    apple_pos = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))
    while apple_pos in snake_body:
        apple_pos = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))

def plot_training_metrics():
    global PLOTTINGFREQ
    window_size = PLOTTINGFREQ
    if len(episode_scores) >= window_size:
        averages_scores = [np.mean(episode_scores[i:i+window_size]) for i in range(0, len(episode_scores), window_size)]
        max_scores = [np.max(episode_scores[i:i+window_size]) for i in range(0, len(episode_scores), window_size)]
        averages_rewards = [np.mean(episode_rewards[i:i+window_size]) for i in range(0, len(episode_rewards), window_size)]
        x_ticks = list(range(0, len(episode_scores), window_size))

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot average score on left y-axis
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Average Score", color='blue')
        ax1.plot(x_ticks, averages_scores, label="Average Score (per {} episodes)".format(window_size), color='blue')
        ax1.plot(x_ticks, max_scores, label="Max Score (per {} episodes)".format(window_size), color='red')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Create second y-axis for average reward
        ax2 = ax1.twinx()
        ax2.set_ylabel("Average Reward", color='green')
        ax2.plot(x_ticks, averages_rewards, label="Average Reward (per {} episodes)".format(window_size), color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        # Title and layout
        plt.title("Training Progress")
        fig.tight_layout()

        # Add legends separately for clarity
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.grid(True)
        plt.savefig("training_metrics.png")
        plt.show()
    else:
        print("Not enough episodes to plot training metrics.")


# === HELPER FUNCTIONS ===
def move_apple():
    global apple_pos
    if TRAINING_MODE:
        while True:
            new_pos = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))
            if new_pos not in snake_body:
                apple_pos = new_pos
                break
    else:
        # Existing GUI apple movement
        while True:
            row = random.randint(0, GRID_HEIGHT - 1)
            col = random.randint(0, GRID_WIDTH - 1)
            x1, y1 = col * BOX_SIZE, row * BOX_SIZE
            overlap = any(canvas.coords(rect)[:2] == [x1, y1] for rect in snake_body)
            if not overlap:
                break
        canvas.coords(apple_rect, x1, y1, x1 + BOX_SIZE, y1 + BOX_SIZE)
        apple_pos = (col, row)

def setup_snake():
    global snake_body, snake_length, action_history, position_history

    direction = "d"  # Reset to initial direction

    if TRAINING_MODE:
        snake_body = [(GRID_WIDTH // 2 - i - 4, GRID_HEIGHT // 2) for i in range(snake_length)]
        initial_head = snake_body[0]
    else:
        snake_body.clear()
        for i in range(snake_length):
            col = GRID_WIDTH // 2 - i - 4
            row = GRID_HEIGHT // 2
            x1 = col * BOX_SIZE
            y1 = row * BOX_SIZE
            rect = canvas.create_rectangle(x1, y1, x1 + BOX_SIZE, y1 + BOX_SIZE, fill="#4287f5", outline="lightgrey")
            snake_body.append(rect)
        initial_head = (GRID_WIDTH // 2 - 4, GRID_HEIGHT // 2)

    # Reset tracking histories
    action_history.clear()
    position_history.clear()
    position_history.append(initial_head)

def get_state():
    if TRAINING_MODE:
        if TRAINING_MODE:
            hc, hr = snake_body[0]
        else:
            coords = canvas.coords(snake_body[0])
            hc = int(coords[0] // BOX_SIZE)
            hr = int(coords[1] // BOX_SIZE)
        ac, ar = apple_pos
    else:
        # Get coordinates from canvas items
        hc = int(canvas.coords(snake_body[0])[0] // BOX_SIZE)
        hr = int(canvas.coords(snake_body[0])[1] // BOX_SIZE)
        ac = int(canvas.coords(apple_rect)[0] // BOX_SIZE)
        ar = int(canvas.coords(apple_rect)[1] // BOX_SIZE)

    rel_walls = [
        (GRID_WIDTH - 1 - hc) if direction == 'd' else hc if direction == 'a' else hr if direction == 'w' else (GRID_HEIGHT - 1 - hr),
        hr if direction == 'd' else (GRID_HEIGHT - 1 - hr) if direction == 'a' else hc if direction == 'w' else (GRID_WIDTH - 1 - hc),
        (GRID_HEIGHT - 1 - hr) if direction == 'd' else hr if direction == 'a' else (GRID_WIDTH - 1 - hc) if direction == 'w' else hc
    ]

    body_dists = {'ahead': float('inf'), 'left': float('inf'), 'right': float('inf')}
    for segment in snake_body[1:]:
        if TRAINING_MODE:
            sc, sr = segment
        else:
            sc = int(canvas.coords(segment)[0] // BOX_SIZE)
            sr = int(canvas.coords(segment)[1] // BOX_SIZE)

        dc, dr = sc - hc, sr - hr
        if direction == 'd':
            if dr == 0 and dc > 0: body_dists['ahead'] = min(body_dists['ahead'], dc)
            if dr < 0 and dc == 0: body_dists['left'] = min(body_dists['left'], -dr)
            if dr > 0 and dc == 0: body_dists['right'] = min(body_dists['right'], dr)
        elif direction == 'a':
            if dr == 0 and dc < 0: body_dists['ahead'] = min(body_dists['ahead'], -dc)
            if dr > 0 and dc == 0: body_dists['left'] = min(body_dists['left'], dr)
            if dr < 0 and dc == 0: body_dists['right'] = min(body_dists['right'], -dr)
        elif direction == 'w':
            if dc == 0 and dr < 0: body_dists['ahead'] = min(body_dists['ahead'], -dr)
            if dc < 0 and dr == 0: body_dists['left'] = min(body_dists['left'], -dc)
            if dc > 0 and dr == 0: body_dists['right'] = min(body_dists['right'], dc)
        elif direction == 's':
            if dc == 0 and dr > 0: body_dists['ahead'] = min(body_dists['ahead'], dr)
            if dc > 0 and dr == 0: body_dists['left'] = min(body_dists['left'], dc)
            if dc < 0 and dr == 0: body_dists['right'] = min(body_dists['right'], -dc)

    max_d = max(GRID_WIDTH, GRID_HEIGHT)
    for k in body_dists:
        if body_dists[k] == float('inf'):
            body_dists[k] = max_d

    apple_fwd = (ac - hc) if direction == 'd' else (hc - ac) if direction == 'a' else (hr - ar) if direction == 'w' else (ar - hr)
    apple_side = (hr - ar) if direction == 'd' else (ar - hr) if direction == 'a' else (ac - hc) if direction == 'w' else (hc - ac)
    return [x / max_d for x in rel_walls + [body_dists['ahead'], body_dists['left'], body_dists['right'], apple_fwd, apple_side]]

def apply_action(action):
    global direction
    turns = {
        0: {'d': 'w', 'w': 'a', 'a': 's', 's': 'd'},
        1: {'d': 'd', 'w': 'w', 'a': 'a', 's': 's'},
        2: {'d': 's', 's': 'a', 'a': 'w', 'w': 'd'}
    }
    direction = turns[action][direction]

def game_over():
    global game_running
    game_running = False
    if not TRAINING_MODE:
        canvas.create_text(GRID_WIDTH * BOX_SIZE // 2, GRID_HEIGHT * BOX_SIZE // 2, text="game Over", fill="red", font=("Arial", 24), tags="game_over")
        restart_button.pack()
    if TRAINING_MODE:
        agent.save_model()

def game_loop():
    global score, move_reward, episode_reward, snake_length, game_running, snake_body, MOVE_LOITER_PENALTY, APPLE_REWARD, COLLISION_PENALTY, LOOP_PENALTY, LOOP_WINDOW, MIN_LOOP_LENGTH, action_history, position_history, pending_loop, highscore
    if not game_running:
        return

    state = get_state()
    action = agent.act(state)
    apply_action(action)

    # Calculate new head position
    if TRAINING_MODE:
        hc, hr = snake_body[0]
    else:
        coords = canvas.coords(snake_body[0])
        hc = int(coords[0] // BOX_SIZE)
        hr = int(coords[1] // BOX_SIZE)
    if direction == 'w': hr -= 1
    elif direction == 's': hr += 1
    elif direction == 'a': hc -= 1
    elif direction == 'd': hc += 1
    new_head = (hc, hr)

    # === Check for collisions ===
    if TRAINING_MODE:
        if new_head in snake_body or hc < 0 or hc >= GRID_WIDTH or hr < 0 or hr >= GRID_HEIGHT:
            move_reward = COLLISION_PENALTY
            episode_reward += move_reward
            agent.remember(state, action, move_reward, get_state(), True)
            agent.train()
            game_over()
            return
        snake_body.insert(0, new_head)
    else:
        if hc < 0 or hc >= GRID_WIDTH or hr < 0 or hr >= GRID_HEIGHT or new_head in [
            (int(canvas.coords(seg)[0] // BOX_SIZE), int(canvas.coords(seg)[1] // BOX_SIZE)) for seg in snake_body
        ]:
            move_reward = COLLISION_PENALTY
            episode_reward += move_reward
            agent.remember(state, action, move_reward, get_state(), True)
            agent.train()
            game_over()
            return
        x1 = hc * BOX_SIZE
        y1 = hr * BOX_SIZE
        rect = canvas.create_rectangle(x1, y1, x1 + BOX_SIZE, y1 + BOX_SIZE, fill="#4287f5", outline="lightgrey")
        snake_body.insert(0, rect)

    # === Check for apple ===
    if new_head == apple_pos:
        score += 1
        if score > highscore:
            highscore = score
        move_reward = APPLE_REWARD
        snake_length += 1
        if not TRAINING_MODE:
            score_label.config(text=f"Score: {score}")
            highscore_label.config(text=f"Highscore: {highscore}")
        move_apple()
    else:
        if len(snake_body) > snake_length:
            if not TRAINING_MODE:
                tail = snake_body.pop()
                canvas.delete(tail)
            else:
                snake_body.pop()
        move_reward = MOVE_LOITER_PENALTY

        # Loop-Detection
        position_history.append(new_head)
        action_history.append(action)

        position_loop_penalty = 0
        if len(position_history) >= LOOP_WINDOW:
            unique_positions = len(set(position_history))
            if unique_positions < LOOP_WINDOW / 3:
                position_loop_penalty = LOOP_PENALTY * (LOOP_WINDOW - unique_positions)

        action_loop_penalty = 0
        if len(action_history) >= MIN_LOOP_LENGTH:
            last_actions = list(action_history)[-MIN_LOOP_LENGTH:]
            if all(a == action for a in last_actions):
                action_loop_penalty = LOOP_PENALTY * MIN_LOOP_LENGTH

        total_penalty = position_loop_penalty + action_loop_penalty
        if total_penalty < 0:
            move_reward += total_penalty
            episode_reward += total_penalty

    # Training only if in training mode
    if TRAINING_MODE:
        new_state = get_state()
        episode_reward += move_reward
        agent.remember(state, action, move_reward, new_state, False)
        agent.train()

    if not TRAINING_MODE and game_running:
        pending_loop = root.after(SPEED, game_loop)


def restart_game():
    global score, snake_length, snake_body, direction, game_running, move_reward, episode_reward, pending_loop
    global action_history, position_history

    if pending_loop:
        root.after_cancel(pending_loop)
        pending_loop = None

    score, snake_length, move_reward, episode_reward = 0, 3, 0, 0
    direction = "d"
    game_running = True
    action_history.clear()
    position_history.clear()

    if not TRAINING_MODE:
        for segment in snake_body:
            canvas.delete(segment)
        canvas.delete("game_over")

    setup_snake()
    move_apple()
    if not TRAINING_MODE:
        score_label.config(text="Score: 0")
    game_loop()



MAX_STEPS = 1000  # Max steps per episode
last_episode_reward = 0


def main():
    global last_episode_reward, episode_reward, score, EPISODES, TARGET_UPDATE_FREQ
    if TRAINING_MODE:
        for episode in range(EPISODES):
            setup_snake()
            move_apple()
            global game_running
            game_running = True
            episode_reward = 0
            steps = 0

            while game_running and steps < MAX_STEPS:
                game_loop()
                steps += 1

            last_score = score
            last_reward = episode_reward

            episode_scores.append(last_score)
            episode_rewards.append(last_reward)

            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

            print(f"Episode {episode} | Score: {last_score} | Steps: {steps} | Reward: {last_reward:.2f} | Epsilon: {agent.epsilon:.4f}")

            restart_game()

            if episode % TARGET_UPDATE_FREQ == 0 and episode > 0:
                agent.update_target_network()

            if episode % 500 == 0 and episode > 0:
                agent.save_model(f"model_ep{episode}.pth")

        agent.save_model()
        print("Training completed.")
        plot_training_metrics()
    else:
        setup_snake()
        move_apple()
        game_loop()
        root.mainloop()

if __name__ == "__main__":
    main()