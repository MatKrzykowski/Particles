# Pygame_visualization.py
#
# Visualization of my brownian motion script using Pygame library.
#
# Changelog:
# 03.11.2016 - script revision

# Import my common file
from BM_common import Particles, Particle

# Libraries import
import pygame
import sys
import pygame.gfxdraw
from pygame.locals import QUIT

# Size of the window
WIDTH, HEIGHT = 800, 600


def draw_particle(self, screen):
    pygame.draw.circle(screen, self.color, self.intpos, self.radius)
    pygame.gfxdraw.aacircle(screen, *self.intpos, self.radius, self.color)

Particle.draw_particle = draw_particle


def draw_energy(screen, particles):
    textSurfaceObj = fontObj.render("Energy: " + str(particles.total_energy),
                                    True, (0, 0, 0))
    textRectObj.topleft = (0, 0)
    screen.blit(textSurfaceObj, textRectObj)


def draw_FPS(screen):
    textSurfaceObj = fontObj.render("FPS: " + str(round(fpsClock.get_fps(), 1)),
                                    True, (0, 0, 0))
    textRectObj.topright = (699, 0)
    screen.blit(textSurfaceObj, textRectObj)

if __name__ == "__main__":
    pygame.init()  # Initialize pygame

    FPS = 60  # Frames per second
    N = 5  # Number of evolution steps per frame
    fpsClock = pygame.time.Clock()  # Clock initialization

    fontsize = 18

    # Prepare the display
    DISPLAYSURF = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
    pygame.display.set_caption('Brownian motion')

    # Generate list of particles
    particles = Particles(3, 5, 2, 200, 0.002)

    # Prepare print of the text
    fontObj = pygame.font.Font('freesansbold.ttf', fontsize)
    textSurfaceObj = fontObj.render('', True, (0, 0, 0))
    textRectObj = textSurfaceObj.get_rect()

    # Draw the simulation
    while True:
        DISPLAYSURF.fill((255, 255, 255))  # Clear the surface

        # Draw all particles
        for item in particles.items:
            item.draw_particle(DISPLAYSURF)

        draw_energy(DISPLAYSURF, particles)  # Write the energy text
        draw_FPS(DISPLAYSURF)  # Write the FPS text

        for i in range(N):
            particles.time_evolution()

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.update()
        fpsClock.tick(FPS)
