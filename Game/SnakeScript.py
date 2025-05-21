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
direction = "d"  # initial right
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

    # Absolute wall distances
    walls = {
        'north': head_row,
        'south': GRID_HEIGHT - 1 - head_row,
        'west': head_col,
        'east': GRID_WIDTH - 1 - head_col
    }

    # Absolute body distances (initialize with wall distances)
    body = {
        'north': walls['north'],
        'south': walls['south'],
        'west': walls['west'],
        'east': walls['east']
    }

    # Check body segments (skip first 2 to ignore immediate neck)
    for segment in snake_body[2:]:
        seg_x, seg_y, _, _ = canvas.coords(segment)
        seg_col = seg_x // BOX_SIZE
        seg_row = seg_y // BOX_SIZE

        # Vertical alignment (same column)
        if seg_col == head_col:
            if seg_row < head_row:  # North
                dist = head_row - seg_row
                if dist < body['north']:
                    body['north'] = dist
            else:  # South
                dist = seg_row - head_row
                if dist < body['south']:
                    body['south'] = dist

        # Horizontal alignment (same row)
        if seg_row == head_row:
            if seg_col < head_col:  # West
                dist = head_col - seg_col
                if dist < body['west']:
                    body['west'] = dist
            else:  # East
                dist = seg_col - head_col
                if dist < body['east']:
                    body['east'] = dist

    return walls, body


def get_apple_position(head_col, head_row):
    apple_col = apple_x1 // BOX_SIZE
    apple_row = apple_y1 // BOX_SIZE
    return {
        'delta_x': apple_col - head_col,
        'delta_y': apple_row - head_row
    }


def update_data_display():
    if not snake_body or not apple_rect:
        return

    head_x, head_y, _, _ = canvas.coords(snake_body[0])
    head_col = head_x // BOX_SIZE
    head_row = head_y // BOX_SIZE

    walls, body = calculate_distances(head_x, head_y)
    apple_pos = get_apple_position(head_col, head_row)

    data_text = (
        f"Heading: {direction.upper()}\n"
        f"Walls → N={walls['north']} S={walls['south']} W={walls['west']} E={walls['east']}\n"
        f"Body → N={body['north']} S={body['south']} W={body['west']} E={body['east']}\n"
        f"Apple → X:{apple_pos['delta_x']} Y:{apple_pos['delta_y']}"
    )
    data_label.config(text=data_text)


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
        font=("Arial", 18),
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

    update_data_display()  # Update sensor data display


    # Calculate new head position
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

    # Wall collision check
    if (new_head_x1 < 0 or new_head_x1 >= BOX_SIZE * GRID_WIDTH or
            new_head_y1 < 0 or new_head_y1 >= BOX_SIZE * GRID_HEIGHT):
        game_over("Game Over! Snake hit the wall.")
        return

    # Body collision check
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
    if (new_head_x1 == apple_x1 and new_head_y1 == apple_y1):
        snake_length += 1
        score += 1
        score_label.config(text=f"Score: {score}")
        move_apple()
    else:
        if len(snake_body) > snake_length:
            old_tail = snake_body.pop()
            canvas.delete(old_tail)

    speed = max(100, 250 - score * 10)
    root.after(speed, move_snake)


def change_direction(event):
    global direction, direction_locked
    if direction_locked or not game_running:
        return
    keys = {"Up": "w", "Down": "s", "Left": "a", "Right": "d"}
    key = keys.get(event.keysym, event.keysym)
    if key in ("w", "a", "s", "d"):
        if (direction == "w" and key != "s") or \
                (direction == "s" and key != "w") or \
                (direction == "a" and key != "d") or \
                (direction == "d" and key != "a"):
            direction = key
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


root.bind("<Key>", change_direction)

# Setup initial state
setup_snake()
apple_x1 = apple_y1 = 0
apple_x2 = apple_y2 = 0
apple_rect = canvas.create_oval(0, 0, BOX_SIZE, BOX_SIZE, fill="red", outline="")
move_apple()
move_snake()

root.mainloop()