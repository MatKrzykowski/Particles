import sys

# Libraries import
import pygame

# Import my common file
from BM_common import Particles
from common import BLACK, WHITE
from config import default_config as config


def draw_energy(screen, objs, font):
    text_render = font.render("Energy: " + str(objs.total_energy), True, BLACK)
    textRectObj.topleft = (0, 0)
    screen.blit(text_render, textRectObj)


def draw_FPS(screen, font):
    text_render = font.render("FPS: " + str(round(fpsClock.get_fps(), 1)), True, BLACK)
    textRectObj.topright = (699, 0)
    screen.blit(text_render, textRectObj)


if __name__ == "__main__":
    pygame.init()  # Initialize pygame

    fpsClock = pygame.time.Clock()  # Clock initialization

    # Prepare the display
    DISPLAYSURF = pygame.display.set_mode((config.width, config.height), 0, 32)
    pygame.display.set_caption("Brownian motion")

    # Generate list of particles
    particles = Particles(2, 10, 2, 500, 0.002)

    # Prepare print of the text
    fontObj = pygame.font.Font("freesansbold.ttf", config.fontsize)
    textSurfaceObj = fontObj.render("", True, BLACK)
    textRectObj = textSurfaceObj.get_rect()

    # Draw the simulation
    while True:
        DISPLAYSURF.fill(WHITE)  # Clear the surface

        # Draw all particles
        for item in particles.items:
            item.draw_particle(DISPLAYSURF)

        # draw_energy(DISPLAYSURF, particles, fontObj)  # Write the energy text
        draw_FPS(DISPLAYSURF, fontObj)  # Write the FPS text

        for i in range(config.n_subframes):
            particles.time_evolution()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.update()
        fpsClock.tick(config.fps)
