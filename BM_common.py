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
        self.g = 0 # TODO: Account for gravity in energy calculation

        self.subgid_size = 100

        # List of particles
        self.items = []

        self.generate_particles()

        assert (WIDTH % self.subgid_size == 0)
        assert (HEIGHT % self.subgid_size == 0)
        self.n_grid_x = WIDTH // self.subgid_size
        self.n_grid_y = HEIGHT // self.subgid_size

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

        subgrid = {}
        for i in range(self.n_grid_x):
            for j in range(self.n_grid_y):
                subgrid[(i, j)] = []

        for item in self.items:
            r = item.radius
            x = 1 / self.subgid_size
            min_x = max(0,             int((item.position[0] - r) * x))
            max_x = min(self.n_grid_x, int((item.position[0] + r) * x) + 1)
            min_y = max(0,             int((item.position[1] - r) * x))
            max_y = min(self.n_grid_y, int((item.position[1] + r) * x) + 1)
            for i in range(min_x, max_x):
                for j in range(min_y, max_y):
                    subgrid[(i, j)].append(item)

        for _, item in subgrid.items():
            self.calc_collisions_subgrid(item)

        # Reflect from walls and other effects
        for item in self.items:
            item.apply_gravity(self.g, self.dt)
            item.reflect()  # Reflect particle from walls

    def calc_collisions_subgrid(self, subgrid):
        if not subgrid:
            return None
        # Prepare distance array
        n = len(subgrid)
        dist_arr = np.array([i.position for i in subgrid])
        # 2D array of distance vectors
        x_arr = dist_arr[:,0]
        x_arr = np.subtract.outer(x_arr, x_arr)
        y_arr = dist_arr[:,1]
        y_arr = np.subtract.outer(y_arr, y_arr)
        dist_arr = np.hypot(x_arr, y_arr)

        radius_arr = np.array([i.radius for i in subgrid])
        radius_arr = np.add.outer(radius_arr, radius_arr)
        radius_arr = radius_arr > dist_arr

        # Assign pairs i,j so that they point to the minima
        for i, row in enumerate(dist_arr):
            for j, dist in enumerate(row):
                if j >= i:
                    break
                if radius_arr[i, j]:
                    simulation_step(subgrid[i], subgrid[j], dist)


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

    def apply_gravity(self, g, dt):
        self.velocity[1] += g * dt

    @property
    def intpos(self):
        """Return position tuple cast onto integers."""
        return (int(self.position[0]), int(self.position[1]))


    def draw_particle(self, screen):
        pygame.draw.circle(screen, self.color, self.intpos, self.radius)
        pygame.gfxdraw.aacircle(screen, *self.intpos, self.radius, self.color)
