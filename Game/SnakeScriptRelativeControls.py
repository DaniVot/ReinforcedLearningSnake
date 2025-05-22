import tkinter as tk
import random

BOX_SIZE = 30
GRID_WIDTH = 17
GRID_HEIGHT = 15

root = tk.Tk()
root.title("Snake")

canvas = tk.Canvas(root, width=BOX_SIZE * GRID_WIDTH, height=BOX_SIZE * GRID_HEIGHT)
canvas.pack()

grid_rects = []
direction_locked = False
score = 0
snake_length = 3
snake_body = []
direction = "d"  # initial direction: right
apple_rect = None
game_running = True

# Score label
score_label = tk.Label(root, text=f"Score: {score}", font=("Arial", 16))
score_label.pack()

# Data label
data_label = tk.Label(root, text="", font=("Arial", 12))
data_label.pack()

# Restart button (hidden initially)
restart_button = tk.Button(root, text="Restart", font=("Arial", 14), command=lambda: restart_game())
restart_button.pack_forget()

# Create grid of rectangles with checkered pattern
for row in range(GRID_HEIGHT):
    row_rects = []
    for col in range(GRID_WIDTH):
        x1 = col * BOX_SIZE
        y1 = row * BOX_SIZE
        x2 = x1 + BOX_SIZE
        y2 = y1 + BOX_SIZE
        color = "white" if (row + col) % 2 == 0 else "#d3ffcc"
        rect = canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
        row_rects.append(rect)
    grid_rects.append(row_rects)


def calculate_distances(head_x, head_y):
    head_col = head_x // BOX_SIZE
    head_row = head_y // BOX_SIZE

    # Wall distances (absolute)
    abs_walls = {
        'left': head_col,
        'right': GRID_WIDTH - 1 - head_col,
        'up': head_row,
        'down': GRID_HEIGHT - 1 - head_row
    }

    # Convert to direction-relative walls
    rel_walls = get_relative_walls(direction, abs_walls)

    # Body distances (direction-relative)
    body_dist = {
        'ahead': float('inf'),
        'left': float('inf'),
        'right': float('inf')
    }

    for segment in snake_body[1:]:  # Skip head
        seg_x, seg_y, _, _ = canvas.coords(segment)
        seg_col = seg_x // BOX_SIZE
        seg_row = seg_y // BOX_SIZE

        # Calculate relative positions
        rel_pos = get_relative_position(
            head_col, head_row,
            seg_col, seg_row,
            direction
        )

        # Update minimum distances
        for direction_type in ['ahead', 'left', 'right']:
            if rel_pos[direction_type] < body_dist[direction_type]:
                body_dist[direction_type] = rel_pos[direction_type]

    # Convert infinite distances to safe values
    max_dist = max(GRID_WIDTH, GRID_HEIGHT)
    for k in body_dist:
        if body_dist[k] == float('inf'):
            body_dist[k] = max_dist

    return rel_walls, body_dist


def get_relative_walls(direction, abs_walls):
    if direction == 'd':  # Right
        return (abs_walls['right'], abs_walls['up'], abs_walls['down'])
    elif direction == 'a':  # Left
        return (abs_walls['left'], abs_walls['down'], abs_walls['up'])
    elif direction == 'w':  # Up
        return (abs_walls['up'], abs_walls['left'], abs_walls['right'])
    elif direction == 's':  # Down
        return (abs_walls['down'], abs_walls['right'], abs_walls['left'])


def get_relative_position(hc, hr, sc, sr, direction):
    # Returns distances in relative directions (ahead, left, right)
    if direction == 'd':  # Right
        ahead_dist = sc - hc if sr == hr and sc > hc else float('inf')
        left_dist = hr - sr if sc == hc and sr < hr else float('inf')
        right_dist = sr - hr if sc == hc and sr > hr else float('inf')
    elif direction == 'a':  # Left
        ahead_dist = hc - sc if sr == hr and sc < hc else float('inf')
        left_dist = sr - hr if sc == hc and sr > hr else float('inf')
        right_dist = hr - sr if sc == hc and sr < hr else float('inf')
    elif direction == 'w':  # Up
        ahead_dist = hr - sr if sc == hc and sr < hr else float('inf')
        left_dist = hc - sc if sr == hr and sc < hc else float('inf')
        right_dist = sc - hc if sr == hr and sc > hc else float('inf')
    elif direction == 's':  # Down
        ahead_dist = sr - hr if sc == hc and sr > hr else float('inf')
        left_dist = sc - hc if sr == hr and sc > hc else float('inf')
        right_dist = hc - sc if sr == hr and sc < hc else float('inf')

    return {
        'ahead': ahead_dist,
        'left': left_dist,
        'right': right_dist
    }


def get_apple_direction(hc, hr, ac, ar, direction):
    # Get apple position relative to current heading
    if direction == 'd':  # Right
        fwd = ac - hc
        side = hr - ar
    elif direction == 'a':  # Left
        fwd = hc - ac
        side = ar - hr
    elif direction == 'w':  # Up
        fwd = hr - ar
        side = ac - hc
    elif direction == 's':  # Down
        fwd = ar - hr
        side = hc - ac
    return fwd, side


def update_data_display():
    if not snake_body or not apple_rect:
        return

    head_x, head_y, _, _ = canvas.coords(snake_body[0])
    hc = head_x // BOX_SIZE
    hr = head_y // BOX_SIZE
    ac = apple_x1 // BOX_SIZE
    ar = apple_y1 // BOX_SIZE

    walls, body = calculate_distances(head_x, head_y)
    apple_fwd, apple_side = get_apple_direction(hc, hr, ac, ar, direction)

    data_text = (
        f"Heading: {direction}\n"
        f"Walls → Ahead={walls[0]} Left={walls[1]} Right={walls[2]}\n"
        f"Body  → Ahead={body['ahead']} Left={body['left']} Right={body['right']}\n"
        f"Apple → Fwd:{apple_fwd} Side:{apple_side}"
    )
    data_label.config(text=data_text)


def setup_snake():
    global snake_body
    snake_body.clear()
    for i in range(snake_length):
        snake_row = GRID_HEIGHT // 2
        snake_col = GRID_WIDTH // 2 - i - 4
        snake_x1 = snake_col * BOX_SIZE
        snake_y1 = snake_row * BOX_SIZE
        snake_x2 = snake_x1 + BOX_SIZE
        snake_y2 = snake_y1 + BOX_SIZE
        rect = canvas.create_rectangle(snake_x1, snake_y1, snake_x2, snake_y2, fill="#4287f5", outline="lightgrey")
        snake_body.append(rect)


def move_apple():
    global apple_rect, apple_x1, apple_y1
    while True:
        apple_row = random.randint(0, GRID_HEIGHT - 1)
        apple_col = random.randint(0, GRID_WIDTH - 1)
        new_x1 = apple_col * BOX_SIZE
        new_y1 = apple_row * BOX_SIZE
        overlap = False
        for rect in snake_body:
            coords = canvas.coords(rect)
            if coords[0] == new_x1 and coords[1] == new_y1:
                overlap = True
                break
        if not overlap:
            break
    apple_x1 = new_x1
    apple_y1 = new_y1
    new_x2 = apple_x1 + BOX_SIZE
    new_y2 = apple_y1 + BOX_SIZE
    canvas.coords(apple_rect, apple_x1, apple_y1, new_x2, new_y2)


def game_over(message):
    global game_running
    game_running = False
    canvas.create_text(
        BOX_SIZE * GRID_WIDTH // 2,
        BOX_SIZE * GRID_HEIGHT // 2,
        text=message,
        fill="red",
        font=("Arial", 24),
        tags="gameover"
    )
    restart_button.pack()


def move_snake():
    global direction_locked, snake_length, score, game_running
    if not game_running:
        return

    direction_locked = False
    global snake_body, direction

    head_rect = snake_body[0]
    head_coords = canvas.coords(head_rect)
    head_x1, head_y1, head_x2, head_y2 = head_coords

    update_data_display()

    # Calculate new head position based on current direction
    if direction == "w":
        new_head_y1 = head_y1 - BOX_SIZE
        new_head_x1 = head_x1
    elif direction == "s":
        new_head_y1 = head_y1 + BOX_SIZE
        new_head_x1 = head_x1
    elif direction == "a":
        new_head_x1 = head_x1 - BOX_SIZE
        new_head_y1 = head_y1
    elif direction == "d":
        new_head_x1 = head_x1 + BOX_SIZE
        new_head_y1 = head_y1

    # Collision checks
    if (new_head_x1 < 0 or new_head_x1 >= BOX_SIZE * GRID_WIDTH or
            new_head_y1 < 0 or new_head_y1 >= BOX_SIZE * GRID_HEIGHT):
        game_over("Game Over! Snake hit the wall.")
        return

    for rect in snake_body:
        coords = canvas.coords(rect)
        if coords[0] == new_head_x1 and coords[1] == new_head_y1:
            game_over("Game Over! Snake collided with itself.")
            return

    # Create new head
    new_head = canvas.create_rectangle(new_head_x1, new_head_y1,
                                       new_head_x1 + BOX_SIZE,
                                       new_head_y1 + BOX_SIZE,
                                       fill="#4287f5", outline="lightgrey")
    snake_body.insert(0, new_head)

    # Apple consumption
    if new_head_x1 == apple_x1 and new_head_y1 == apple_y1:
        snake_length += 1
        score += 1
        score_label.config(text=f"Score: {score}")
        move_apple()
    else:
        if len(snake_body) > snake_length:
            old_tail = snake_body.pop()
            canvas.delete(old_tail)

    speed = 250
    root.after(speed, move_snake)


def change_direction(event):
    global direction, direction_locked
    if direction_locked or not game_running:
        return

    # Relative turning controls
    turn_map = {
        'Left': {
            'd': 'w',  # Right -> Up
            'w': 'a',  # Up -> Left
            'a': 's',  # Left -> Down
            's': 'd'  # Down -> Right
        },
        'Right': {
            'd': 's',  # Right -> Down
            's': 'a',  # Down -> Left
            'a': 'w',  # Left -> Up
            'w': 'd'  # Up -> Right
        }
    }

    if event.keysym in ['Left', 'Right']:
        new_dir = turn_map[event.keysym][direction]
        if (new_dir == 'w' and direction != 's') or \
                (new_dir == 's' and direction != 'w') or \
                (new_dir == 'a' and direction != 'd') or \
                (new_dir == 'd' and direction != 'a'):
            direction = new_dir
            direction_locked = True


def restart_game():
    global snake_body, snake_length, direction, score, game_running
    for rect in snake_body:
        canvas.delete(rect)
    snake_body.clear()
    canvas.delete("gameover")
    snake_length = 3
    direction = "d"
    score = 0
    score_label.config(text=f"Score: {score}")
    data_label.config(text="")
    restart_button.pack_forget()
    game_running = True
    setup_snake()
    move_apple()
    move_snake()


root.bind("<Left>", change_direction)
root.bind("<Right>", change_direction)

# Setup initial state
setup_snake()
apple_x1 = apple_y1 = 0
apple_x2 = apple_y2 = 0
apple_rect = canvas.create_oval(0, 0, BOX_SIZE, BOX_SIZE, fill="red", outline="")
move_apple()
move_snake()

root.mainloop()