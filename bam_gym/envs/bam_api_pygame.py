import pygame
import numpy as np
import cv2

def show_images_in_grid(images: list[np.ndarray], titles: list[str], status_text: str):
    pygame.init()
    tile_width, tile_height = 240, 180
    text_bar_height = 40
    grid_cols = 2
    grid_rows = 2

    window_width = tile_width * grid_cols
    window_height = tile_height * grid_rows + text_bar_height
    window = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Image Grid Viewer")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 20)

    # Main loop (just render one frame)
    running = True
    while running:
        window.fill((0, 0, 0))  # Clear screen

        for i, img in enumerate(images):
            if img is None:
                continue
            # Resize image to tile size
            img = cv2.resize(img, (tile_width, tile_height))
            img = img.swapaxes(0, 1)  # PyGame expects (W, H, 3)
            surface = pygame.surfarray.make_surface(img)

            row = i // grid_cols
            col = i % grid_cols
            x = col * tile_width
            y = row * tile_height
            window.blit(surface, (x, y))

            # Draw title on top of each tile
            if i < len(titles):
                title_surface = font.render(titles[i], True, (255, 255, 255))
                window.blit(title_surface, (x + 5, y + 5))

        # Draw text bar at bottom
        pygame.draw.rect(window, (255, 255, 255), (0, tile_height * grid_rows, window_width, text_bar_height))
        text_surface = font.render(status_text, True, (0, 0, 0))
        window.blit(text_surface, (10, tile_height * grid_rows + 10))

        pygame.display.update()
        clock.tick(30)

        # Exit after 1 frame (or wait for quit if you want interaction)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()


img1 = np.ones((480, 640, 3), dtype=np.uint8) * 255  # White image
img2 = np.zeros((480, 640, 3), dtype=np.uint8)       # Black image
img3 = np.full((480, 640, 3), (0, 0, 255), dtype=np.uint8)  # Red
img4 = np.full((480, 640, 3), (0, 255, 0), dtype=np.uint8)  # Green

show_images_in_grid(
    images=[img1, img2, img3, img4],
    titles=["Cam1", "Cam2", "Mask", "Depth"],
    status_text="Status: OK | FPS: 30 | Env: bam_env"
)
