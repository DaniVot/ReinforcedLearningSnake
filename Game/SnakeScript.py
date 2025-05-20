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

# Score label
score_label = tk.Label(root, text=f"Score: {score}", font=("Arial", 16))
score_label.pack()

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

# Create snake
snake_length = 3
snake_body = []
for i in range(snake_length):
    snake_row = GRID_HEIGHT // 2
    snake_col = GRID_WIDTH // 2 - i - 4
    snake_x1 = snake_col * BOX_SIZE
    snake_y1 = snake_row * BOX_SIZE
    snake_x2 = snake_x1 + BOX_SIZE
    snake_y2 = snake_y1 + BOX_SIZE
    rect = canvas.create_rectangle(snake_x1, snake_y1, snake_x2, snake_y2, fill="#4287f5", outline="lightgrey")
    snake_body.append(rect)

direction = "d"  # Initial direction: right

# Place apple initially in the middle
apple_row = GRID_HEIGHT // 2
apple_col = GRID_WIDTH // 2
apple_x1 = apple_col * BOX_SIZE
apple_y1 = apple_row * BOX_SIZE
apple_x2 = apple_x1 + BOX_SIZE
apple_y2 = apple_y1 + BOX_SIZE
apple_rect = canvas.create_oval(apple_x1, apple_y1, apple_x2, apple_y2, fill="red", outline="")


def move_apple():
    global apple_row, apple_col, apple_x1, apple_y1
    while True:
        apple_row = random.randint(0, GRID_HEIGHT - 1)
        apple_col = random.randint(0, GRID_WIDTH - 1)
        new_x1 = apple_col * BOX_SIZE
        new_y1 = apple_row * BOX_SIZE

        # Make sure apple doesn't spawn on the snake:
        overlap = False
        for rect in snake_body:
            coords = canvas.coords(rect)
            if coords[0] == new_x1 and coords[1] == new_y1:
                overlap = True
                break
        if not overlap:
            break

    apple_x1 = new_x1  # update apple position globals!
    apple_y1 = new_y1

    new_x2 = apple_x1 + BOX_SIZE
    new_y2 = apple_y1 + BOX_SIZE
    canvas.coords(apple_rect, apple_x1, apple_y1, new_x2, new_y2)


def move_snake():
    global direction_locked, snake_length, score
    direction_locked = False  # Allow direction change again
    global snake_body, direction

    # Get current head position (snake_body[0] is head)
    head_rect = snake_body[0]
    head_coords = canvas.coords(head_rect)
    head_x1, head_y1, head_x2, head_y2 = head_coords

    # Calculate new head position based on direction
    if direction == "w":
        new_head_y1 = head_y1 - BOX_SIZE
        new_head_y2 = head_y2 - BOX_SIZE
        new_head_x1 = head_x1
        new_head_x2 = head_x2
    elif direction == "s":
        new_head_y1 = head_y1 + BOX_SIZE
        new_head_y2 = head_y2 + BOX_SIZE
        new_head_x1 = head_x1
        new_head_x2 = head_x2
    elif direction == "a":
        new_head_x1 = head_x1 - BOX_SIZE
        new_head_x2 = head_x2 - BOX_SIZE
        new_head_y1 = head_y1
        new_head_y2 = head_y2
    elif direction == "d":
        new_head_x1 = head_x1 + BOX_SIZE
        new_head_x2 = head_x2 + BOX_SIZE
        new_head_y1 = head_y1
        new_head_y2 = head_y2

    # Check boundaries
    if new_head_x1 < 0 or new_head_x2 > BOX_SIZE * GRID_WIDTH or new_head_y1 < 0 or new_head_y2 > BOX_SIZE * GRID_HEIGHT:
        print("Game Over! Snake hit the wall.")
        return  # stop moving

    # Check self collision
    for rect in snake_body:
        coords = canvas.coords(rect)
        if coords[0] == new_head_x1 and coords[1] == new_head_y1:
            print("Game Over! Snake collided with itself.")
            return

    # Insert new head at new position (create new rectangle)
    new_head_rect = canvas.create_rectangle(new_head_x1, new_head_y1, new_head_x2, new_head_y2, fill="#4287f5", outline="lightgrey")
    snake_body.insert(0, new_head_rect)

    # Check if snake ate the apple
    if new_head_x1 == apple_x1 and new_head_y1 == apple_y1:
        snake_length += 1
        score += 1
        score_label.config(text=f"Score: {score}")
        move_apple()  # move apple to new spot
    else:
        # Remove tail if no eating (keep length)
        if len(snake_body) > snake_length:
            tail_rect = snake_body.pop()
            canvas.delete(tail_rect)

    # Schedule next move
    root.after(250, move_snake)

def change_direction(event):
    global direction, direction_locked
    if direction_locked:
        return  # Ignore key presses if direction is locked

    if event.keysym in ("w", "a", "s", "d"):
        # Prevent the snake from reversing direction
        if (direction == "w" and event.keysym != "s") or \
           (direction == "s" and event.keysym != "w") or \
           (direction == "a" and event.keysym != "d") or \
           (direction == "d" and event.keysym != "a"):
            direction = event.keysym
            direction_locked = True

root.bind("<Key>", change_direction)

# Start moving snake
move_snake()

root.mainloop()
