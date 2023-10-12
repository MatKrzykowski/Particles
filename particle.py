"""particle.py"""

import numpy as np
import pygame
import pygame.gfxdraw

from config import default_config as config

from common import PURPLE, length, sqlength

class Particle:
    """Class describing particle"""
    rho = 1.0  # Density

    def __init__(self, x, y, vx, vy, radius, color=PURPLE):
        """Instantiation operation.

        x, y - position [pixel], integer,
        vx, vy - velosity [pixel/s], double,
        radius - radius [pixel], integer,
        color - color, tuple of 3 integers between 0-255.
        """
        # Cartesian coordinates tuple
        self.position = np.array([float(x), float(y)], dtype="float64")
        # Velocity vector tuple
        self.velocity = np.array([float(vx), float(vy)], dtype="float64")
        self.mass = self.rho * np.pi * radius**2  # Particle mass
        self.radius = radius  # Particle radius
        self.color = color  # Particle color

        self.min_x = None
        self.max_x = None
        self.min_y = None
        self.max_y = None

    @property
    def kinetic(self):
        """Kinetic energy of the particle.

        Returns kinetic energy, double."""
        return self.mass * sqlength(self.velocity) / 2.0

    @property
    def momentum(self):
        """Momentum value of the particle.

        Returns momentum, double."""
        return self.mass * length(self.velocity)

    def increment(self, dt):
        """Change position due to velocity."""
        self.position += self.velocity * dt

    def reflect(self):
        """Bouncing off the walls by the particle."""
        # Along the x axis
        if self.position[0] > config.width - self.radius and self.velocity[
                0] > 0:
            self.velocity[0] *= -1
        elif self.position[0] < self.radius and self.velocity[0] < 0:
            self.velocity[0] *= -1
        # Along the y axis
        if self.position[1] > config.height - self.radius and self.velocity[
                1] > 0:
            self.velocity[1] *= -1
        elif self.position[1] < self.radius and self.velocity[1] < 0:
            self.velocity[1] *= -1

    def apply_gravity(self, g, dt):
        self.velocity[1] += g * dt

    @property
    def intpos(self):
        """Return position tuple cast onto integers."""
        return (int(self.position[0]), int(self.position[1]))

    def draw_particle(self, screen):
        pygame.draw.circle(screen, self.color, self.intpos, self.radius)
        pygame.gfxdraw.aacircle(screen, *self.intpos, self.radius, self.color)