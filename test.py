import torch
import torch.nn.functional as F
import pygame
import numpy as np
from scipy.ndimage import gaussian_filter
from train import BasicFeedForward

pygame.init()

WINDOW_SIZE = 560
GRID_SIZE = 28
CELL_SIZE = WINDOW_SIZE // GRID_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)

screen = pygame.display.set_mode((WINDOW_SIZE + 200, WINDOW_SIZE))
pygame.display.set_caption("Digit Classifier - Draw a digit")

grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

model = BasicFeedForward()
try:
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    model_loaded = True
except FileNotFoundError:
    print("Model not found. Please train the model first.")
    model_loaded = False

def draw_grid():
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            color_value = int(grid[i, j] * 255)
            color = (255 - color_value, 255 - color_value, 255 - color_value)
            pygame.draw.rect(screen, color, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(screen, GRAY, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

def draw_ui():
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)

    pygame.draw.rect(screen, WHITE, (WINDOW_SIZE, 0, 200, WINDOW_SIZE))

    title = font.render("Prediction:", True, BLACK)
    screen.blit(title, (WINDOW_SIZE + 10, 20))

    if model_loaded:
        with torch.no_grad():
            # Apply Gaussian blur to smooth the drawing
            smoothed_grid = gaussian_filter(grid, sigma=1.0)

            # Normalize using MNIST statistics (same as training)
            input_tensor = torch.from_numpy(smoothed_grid.flatten()).unsqueeze(0)
            input_tensor = (input_tensor - 0.1307) / 0.3081

            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)[0]

            for i in range(10):
                prob = probabilities[i].item()
                bar_width = int(prob * 150)

                label = small_font.render(f"{i}:", True, BLACK)
                screen.blit(label, (WINDOW_SIZE + 10, 70 + i * 40))

                pygame.draw.rect(screen, (100, 200, 100), (WINDOW_SIZE + 40, 75 + i * 40, bar_width, 20))

                pct = small_font.render(f"{prob*100:.1f}%", True, BLACK)
                screen.blit(pct, (WINDOW_SIZE + 40, 75 + i * 40))
    else:
        error_text = small_font.render("Model not loaded", True, (255, 0, 0))
        screen.blit(error_text, (WINDOW_SIZE + 10, 70))

    inst_font = pygame.font.Font(None, 20)
    instructions = [
        "Left click: Draw",
        "Right click: Erase",
        "C: Clear canvas"
    ]
    for i, inst in enumerate(instructions):
        text = inst_font.render(inst, True, BLACK)
        screen.blit(text, (WINDOW_SIZE + 10, WINDOW_SIZE - 100 + i * 25))

def paint_cell(pos, erase=False):
    x, y = pos
    if x < WINDOW_SIZE:
        grid_x = y // CELL_SIZE
        grid_y = x // CELL_SIZE

        if 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
            if erase:
                grid[grid_x, grid_y] = 0
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = grid_x + dx, grid_y + dy
                        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                            grid[nx, ny] = 0
            else:
                grid[grid_x, grid_y] = 1
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = grid_x + dx, grid_y + dy
                        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                            grid[nx, ny] = max(grid[nx, ny], 0.8 if (dx != 0 or dy != 0) else 1)

def clear_grid():
    global grid
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

running = True
drawing = False
erasing = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                drawing = True
            elif event.button == 3:
                erasing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                drawing = False
            elif event.button == 3:
                erasing = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                clear_grid()

    if drawing or erasing:
        pos = pygame.mouse.get_pos()
        paint_cell(pos, erase=erasing)

    screen.fill(WHITE)
    draw_grid()
    draw_ui()
    pygame.display.flip()

    pygame.time.Clock().tick(60)

pygame.quit()
