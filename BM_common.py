# BM_common.py
#
# Basis for Brownian motion simulation.
#
# Based on program I've written in AngularJS in 2014
# https://www.khanacademy.org/computer-programming/elastic-collisionsbrownian-movement/5114294751461376
#
# Changelog:
# 18.09.2017 - Script updated and revised
# 19.09.2017 - Functions collected into Particles class
#
# To-do:
# * Damping
# * Gravity
# * Statistics for thermodynamics

# Libraries imports
import numpy as np  # Import math modules

import pygame

# Size of the window
WIDTH, HEIGHT = 800, 600

# Colors
BLACK = (0, 0, 0)
BLUE = (0, 128, 255)
RED = (255, 0, 0)
PURPLE = (200, 0, 255)


def sqlength(A):
    """Find square length of a 2-element vector A.

    A - vector, numpyarray.
    """
    return np.sum(A**2)


def length(A):
    """Find length of a 2-element vector A.

    A - vector, numpyarray.
    """
    return np.linalg.norm(A)


def simulation_step(i, j, dist):
    """Simulation step for dwo particles and their distance.

    i, j - particles,
    dist - distance between i and j, double.
    """
    if dist <= i.radius + j.radius:  # Collision if distance too small
        # Scalar product appearing in the calculations
        i_pos, i_vel = i.position, i.velocity
        j_pos, j_vel = j.position, j.velocity
        scalar = np.sum((i_pos - j_pos) * (i_vel - j_vel))
        if scalar < 0.0:  # Check if collision didn't already happen
            # Calculation of constant appearing in the calculations
            A = 2 * scalar / (i.mass + j.mass) / dist**2
            # Change in the velocity vectors for i and j
            i.velocity, j.velocity = \
                i_vel - j.mass * A * (i_pos - j_pos), \
                j_vel - i.mass * A * (j_pos - i_pos)


##########################################################################


class Particles():
    """Class describing all particles and their interactions in the simulation."""

    def __init__(self, min_r, max_r, n_big, n, dt):
        self.min_r = min_r
        self.max_r = max_r
        self.n = n
        self.n_big = n_big
        self.dt = dt

        # List of particles
        self.items = []

        # Matrix with infinite diagonal elements
        self.inf_triag = np.finfo(np.float64).max / 2 * np.tri(
            n, n, dtype="float64").transpose()

        self.generate_particles()

    def generate_big(self):
        """Add up to 2 big particles in specified locations."""
        if self.n_big >= 1:
            self.items.append(Particle(100, 100, 200, 300, 100, color=RED))
        if self.n_big >= 2:
            self.items.append(Particle(500, 100, 100, 300, 100, color=BLUE))

    def generate_small(self):
        """Add up to n small particles in random locations."""
        while len(self.items) < self.n:  # While particles are being generated

            # Randomize radius
            r = np.random.randint(self.min_r, self.max_r + 1)

            # Randomize position
            pos = (r + np.random.randint(WIDTH - 2 * r),
                   r + np.random.randint(HEIGHT - 2 * r))

            # Check if there are no particle overlap
            for item in self.items:
                if length(np.array(pos) - item.position) <= r + item.radius:
                    break

            # If not add new particle to the list
            else:
                self.items.append(Particle(*pos, 0, 0, r))

    def generate_particles(self):
        """Generate n particles, some in random locations."""
        self.generate_big()  # Add 2 large particles in specified locations

        self.generate_small()  # Generate the rest of the particles

        # Assign list of radii of possible collisions
        self.max_dist = np.array([i.radius + self.max_r for i in self.items])

    @property
    def total_energy(self):
        """Calculate sum of all particles' energy.

        Returns total energy of the particles, double.
        """
        return sum(item.kinetic for item in self.items)

    def time_evolution(self):
        """Time evolution method for particles."""

        # Move particles
        for item in self.items:
            item.increment(self.dt)  # Integrate position using velocity

        # Prepare distance array
        dist_arr = np.array([i.position for i in self.items])

        # 2D array of distance vectors
        dist_arr = dist_arr.reshape(self.n, 1, 2) - dist_arr
        dist_arr = np.hypot(dist_arr[:, :, 0], dist_arr[:, :, 1])

        dist_arr += self.inf_triag  # Make diagonal elements, otherwise 0, infinite

        if self.n_big >= 2:  # Unique interaction of 2 big particles
            simulation_step(self.items[0], self.items[1], dist_arr[1, 0])

            # Remaining interactions
            dist_arr = dist_arr[2:, :]  # Remove unnecessary elements

        min_arg = np.argmin(dist_arr, 0)  # Find arguments for minima
        # Assign pairs i,j so that they point to the minima
        for i, j in enumerate(min_arg):
            # Simulation step for found pair of particles
            simulation_step(self.items[i], self.items[j + self.n_big],
                            dist_arr[j, i])

        # Reflect from walls and other effects
        for item in self.items:
            item.reflect()  # Reflect particle from walls


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
        if self.position[0] > WIDTH - self.radius and self.velocity[0] > 0:
            self.velocity[0] *= -1
        elif self.position[0] < self.radius and self.velocity[0] < 0:
            self.velocity[0] *= -1
        # Along the y axis
        if self.position[1] > HEIGHT - self.radius and self.velocity[1] > 0:
            self.velocity[1] *= -1
        elif self.position[1] < self.radius and self.velocity[1] < 0:
            self.velocity[1] *= -1

    @property
    def intpos(self):
        """Return position tuple cast onto integers."""
        return (int(self.position[0]), int(self.position[1]))


    def draw_particle(self, screen):
        pygame.draw.circle(screen, self.color, self.intpos, self.radius)
        pygame.gfxdraw.aacircle(screen, *self.intpos, self.radius, self.color)
