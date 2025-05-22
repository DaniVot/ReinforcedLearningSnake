import tkinter as tk
import random

# === CONFIG ===
BOX_SIZE = 30
GRID_WIDTH = 17
GRID_HEIGHT = 15
TRAINING_MODE = False  # Set to True to fast-forward and skip visuals
SPEED = 10 if TRAINING_MODE else 250

# === STATE ===
move_reward = 0
episode_reward = 0
score = 0
snake_length = 3
snake_body = []
direction = "d"  # right
direction_locked = False
game_running = True

# === TK INTERFACE ===
root = tk.Tk()
root.title("Snake")
canvas = tk.Canvas(root, width=BOX_SIZE * GRID_WIDTH, height=BOX_SIZE * GRID_HEIGHT)
canvas.pack()
score_label = tk.Label(root, text=f"Score: {score}", font=("Arial", 16))
score_label.pack()
data_label = tk.Label(root, text="", font=("Arial", 12))
data_label.pack()
restart_button = tk.Button(root, text="Restart", font=("Arial", 14), command=lambda: restart_game())
restart_button.pack_forget()

# === GRID DRAWING ===
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

# === APPLE ===
apple_rect = canvas.create_oval(0, 0, BOX_SIZE, BOX_SIZE, fill="red", outline="")

def move_apple():
    while True:
        row = random.randint(0, GRID_HEIGHT - 1)
        col = random.randint(0, GRID_WIDTH - 1)
        x1, y1 = col * BOX_SIZE, row * BOX_SIZE
        overlap = any(canvas.coords(rect)[:2] == [x1, y1] for rect in snake_body)
        if not overlap:
            break
    canvas.coords(apple_rect, x1, y1, x1 + BOX_SIZE, y1 + BOX_SIZE)

# === SETUP ===
def setup_snake():
    global snake_body
    snake_body.clear()
    for i in range(snake_length):
        row = GRID_HEIGHT // 2
        col = GRID_WIDTH // 2 - i - 4
        x1 = col * BOX_SIZE
        y1 = row * BOX_SIZE
        rect = canvas.create_rectangle(x1, y1, x1 + BOX_SIZE, y1 + BOX_SIZE, fill="#4287f5", outline="lightgrey")
        snake_body.append(rect)

def restart_game():
    global score, snake_length, direction, game_running, move_reward, episode_reward
    for rect in snake_body:
        canvas.delete(rect)
    snake_body.clear()
    canvas.delete("gameover")
    snake_length = 3
    direction = "d"
    score = 0
    move_reward = 0
    episode_reward = 0
    game_running = True
    score_label.config(text=f"Score: {score}")
    data_label.config(text="")
    restart_button.config(state="disabled")
    restart_button.pack_forget()
    setup_snake()
    move_apple()
    game_loop()

# === STATE ===
def get_state():
    head_x, head_y, _, _ = canvas.coords(snake_body[0])
    hc, hr = int(head_x) // BOX_SIZE, int(head_y) // BOX_SIZE
    ax1, ay1, _, _ = canvas.coords(apple_rect)
    ac, ar = int(ax1) // BOX_SIZE, int(ay1) // BOX_SIZE

    rel_walls = [
        (GRID_WIDTH - 1 - hc) if direction == 'd' else hc if direction == 'a' else hr if direction == 'w' else (GRID_HEIGHT - 1 - hr),
        hr if direction == 'd' else (GRID_HEIGHT - 1 - hr) if direction == 'a' else hc if direction == 'w' else (GRID_WIDTH - 1 - hc),
        (GRID_HEIGHT - 1 - hr) if direction == 'd' else hr if direction == 'a' else (GRID_WIDTH - 1 - hc) if direction == 'w' else hc
    ]

    body_dists = {'ahead': float('inf'), 'left': float('inf'), 'right': float('inf')}
    for segment in snake_body[1:]:
        sx, sy, _, _ = canvas.coords(segment)
        sc, sr = int(sx) // BOX_SIZE, int(sy) // BOX_SIZE
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

def ai_decision(state):
    # Dummy policy (replace with RL model)
    _, _, _, _, _, _, af, asd = state
    if asd < -0.1:
        return 0  # left
    elif asd > 0.1:
        return 2  # right
    else:
        return 1  # forward

def apply_action(action):
    global direction
    turns = {
        0: {'d': 'w', 'w': 'a', 'a': 's', 's': 'd'},
        1: {'d': 'd', 'w': 'w', 'a': 'a', 's': 's'},
        2: {'d': 's', 's': 'a', 'a': 'w', 'w': 'd'}
    }
    direction = turns[action][direction]

def game_over(msg):
    global game_running
    game_running = False
    canvas.create_text(GRID_WIDTH * BOX_SIZE // 2, GRID_HEIGHT * BOX_SIZE // 2, text=msg, fill="red",
                       font=("Arial", 24), tags="gameover")
    restart_button.config(state="normal")
    restart_button.pack()

def game_loop():
    global move_reward, episode_reward, score, snake_length
    if not game_running:
        return

    update_data_display()
    move_reward = 0

    state = get_state()
    action = ai_decision(state)
    apply_action(action)

    hx1, hy1, _, _ = canvas.coords(snake_body[0])
    if direction == 'w':
        hy1 -= BOX_SIZE
    elif direction == 's':
        hy1 += BOX_SIZE
    elif direction == 'a':
        hx1 -= BOX_SIZE
    elif direction == 'd':
        hx1 += BOX_SIZE

    if hx1 < 0 or hx1 >= BOX_SIZE * GRID_WIDTH or hy1 < 0 or hy1 >= BOX_SIZE * GRID_HEIGHT:
        move_reward = -1
        episode_reward += move_reward
        update_data_display()
        game_over("Wall Hit")
        return

    for rect in snake_body:
        if canvas.coords(rect)[:2] == [hx1, hy1]:
            move_reward = -1
            episode_reward += move_reward
            update_data_display()
            game_over("Self Collision")
            return

    new_head = canvas.create_rectangle(hx1, hy1, hx1 + BOX_SIZE, hy1 + BOX_SIZE, fill="#4287f5", outline="lightgrey")
    snake_body.insert(0, new_head)

    ax1, ay1, _, _ = canvas.coords(apple_rect)
    if hx1 == ax1 and hy1 == ay1:
        score += 1
        snake_length += 1
        move_reward = 1
        score_label.config(text=f"Score: {score}")
        move_apple()
    else:
        if len(snake_body) > snake_length:
            tail = snake_body.pop()
            canvas.delete(tail)

    if move_reward == 0:
        move_reward = -0.1

    episode_reward += move_reward

    if TRAINING_MODE:
        game_loop()
    else:
        root.after(SPEED, game_loop)

def update_data_display():
    state = get_state()
    state_str = "\n".join([f"{v:.3f}" for v in state])
    data_label.config(
        text=f"State:\n{state_str}\nReward (step): {move_reward}\nReward (total): {episode_reward:.2f}"
    )

def change_direction(event):
    global direction, direction_locked
    if direction_locked or not game_running:
        return
    if event.keysym == 'Left':
        apply_action(0)
    elif event.keysym == 'Right':
        apply_action(2)
    direction_locked = True

# === MAIN ===
def main():
    setup_snake()
    move_apple()
    game_loop()
    root.bind("<Left>", change_direction)
    root.bind("<Right>", change_direction)
    root.mainloop()

main()