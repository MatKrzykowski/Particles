"""BM_common.py"""

import numpy as np

from common import RED, BLUE, length
from config import default_config as config
from particle import Particle
from physics import elastic_collision
from subgrid import Subgrid


class Particles():
    """Class describing all particles and their interactions in the simulation."""

    def __init__(self, min_r, max_r, n_big, n, dt):
        self.min_r = min_r
        self.max_r = max_r
        self.n = n
        self.n_big = n_big
        self.dt = dt
        # self.g = 0

        self.subgid_size = config.subgid_size

        # List of particles
        self.items = []

        self.generate_particles()

        assert config.width % self.subgid_size == 0
        assert config.height % self.subgid_size == 0
        self.n_grid_x = config.width // self.subgid_size
        self.n_grid_y = config.height // self.subgid_size

        self.subgrid = self.generate_subgrid()

    def generate_subgrid(self):
        subgrid = {}
        for i in range(self.n_grid_x):
            for j in range(self.n_grid_y):
                subgrid[(i, j)] = Subgrid()

        for item in self.items:
            r = item.radius
            x = 1 / self.subgid_size
            min_x = max(0, int((item.position[0] - r) * x))
            max_x = min(self.n_grid_x, int((item.position[0] + r) * x) + 1)
            min_y = max(0, int((item.position[1] - r) * x))
            max_y = min(self.n_grid_y, int((item.position[1] + r) * x) + 1)
            item.min_x = min_x
            item.max_x = max_x
            item.min_y = min_y
            item.max_y = max_y
            for i in range(min_x, max_x):
                for j in range(min_y, max_y):
                    subgrid[(i, j)].add(item)
        return subgrid

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
            pos = (r + np.random.randint(config.width - 2 * r),
                   r + np.random.randint(config.height - 2 * r))

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

        z = 1 / self.subgid_size
        for item in self.items:
            self.update_subgrids(item, z)

        for _, item in self.subgrid.items():
            self.calc_collisions_subgrid(item)

        # Reflect from walls and other effects
        for item in self.items:
            # item.apply_gravity(self.g, self.dt)
            item.reflect()

    def update_subgrids(self, item, z):
        """Upgrades subgrids in between frames.
        
        Works assuming a particle won't jump over any subgrid!
        """
        r = item.radius
        min_x = max(0, int((item.position[0] - r) * z))
        max_x = min(self.n_grid_x, int((item.position[0] + r) * z) + 1)
        min_y = max(0, int((item.position[1] - r) * z))
        max_y = min(self.n_grid_y, int((item.position[1] + r) * z) + 1)
        if min_x != item.min_x:
            if min_x > item.min_x:
                for y in range(item.min_y, item.max_y):
                    self.subgrid[(item.min_x, y)].remove(item)
            else:
                for y in range(item.min_y, item.max_y):
                    self.subgrid[(min_x, y)].add(item)
            item.min_x = min_x
        if max_x != item.max_x:
            if max_x < item.max_x:
                for y in range(item.min_y, item.max_y):
                    self.subgrid[(max_x, y)].remove(item)
            else:
                for y in range(item.min_y, item.max_y):
                    self.subgrid[(item.max_x, y)].add(item)
            item.max_x = max_x
        if min_y != item.min_y:
            if min_y > item.min_y:
                for x in range(item.min_x, item.max_x):
                    self.subgrid[(x, item.min_y)].remove(item)
            else:
                for x in range(item.min_x, item.max_x):
                    self.subgrid[(x, min_y)].add(item)
            item.min_y = min_y
        if max_y != item.max_y:
            if max_y < item.max_y:
                for x in range(item.min_x, item.max_x):
                    self.subgrid[(x, max_y)].remove(item)
            else:
                for x in range(item.min_x, item.max_x):
                    self.subgrid[(x, item.max_y)].add(item)
            item.max_y = max_y

    def calc_collisions_subgrid(self, subgrid):
        if not subgrid:
            return

        dist_arr, applicable_arr = subgrid.dist_arr

        # Assign pairs i,j so that they point to the minima
        for i, row in enumerate(dist_arr):
            for j, dist in enumerate(row):
                if j >= i:
                    break
                if applicable_arr[i, j]:
                    elastic_collision(subgrid.items[i], subgrid.items[j], dist)
