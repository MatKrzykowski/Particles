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

# Size of the window
WIDTH, HEIGHT = 800, 600

# Colors
PURPLE = (0, 128, 255)
RED = (255, 0, 0)


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
        scalar = np.sum((i.position - j.position) * (i.velocity - j.velocity))
        if scalar < 0.0:  # Check if collision didn't already happen
            # Calculation of constant appearing in the calculations
            A = 2 * scalar / (i.mass + j.mass) / dist**2
            # Change in the velocity vectors for i and j
            i.velocity, j.velocity = \
                i.velocity - j.mass * A * (i.position - j.position), \
                j.velocity - i.mass * A * (j.position - i.position)


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
        self.inf_diag = np.diag(np.inf * np.ones(n, dtype="float64"))

        self.generate_particles()

    def generate_big(self):
        """Add up to 2 big particles in specified locations."""
        if self.n_big >= 1:
            self.items.append(Particle(100, 100, 200, 300, 100, color=RED))
        if self.n_big >= 2:
            self.items.append(Particle(500, 100, 100, 300, 100, color=PURPLE))

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
        return sum([item.kinetic for item in self.items])

    def time_evolution(self):
        """Time evolution method for particles."""

        # Move particles
        for item in self.items:
            item.increment(self.dt)  # Integrate position using velocity

        # Prepare distance array
        dist_arr = np.array([i.position for i in self.items])

        # 2D array of distance vectors
        dist_arr = dist_arr.reshape(self.n, 1, 2) - dist_arr
        dist_arr = np.sqrt((dist_arr**2).sum(2))

        dist_arr += self.inf_diag  # Make diagonal elements, otherwise 0, infinite

        if self.n_big >= 2:  # Unique interaction of 2 big particles
            simulation_step(self.items[0], self.items[1], dist_arr[0, 1])

            # Remaining interactions
            dist_arr = dist_arr[2:, :]  # Remove unnecessary elements

        while True:  # Remaining interactions if available
            min_arg = np.argmin(dist_arr, 0)  # Find arguments for minima
            # Assign pairs i,j so that they point to the minima
            for i, j in enumerate(min_arg):
                # Simulation step for found pair of particles
                simulation_step(self.items[i], self.items[j + self.n_big],
                                dist_arr[j, i])
                # Make visited values infinite
                if i < len(min_arg) - self.n_big:
                    dist_arr[i, j] = np.inf
                dist_arr[j, i] = np.inf

            # Break loop if all remaining distance larger than max interaction
            # dist
            min_val = np.min(dist_arr, 0)
            if (min_val >= self.max_dist[0:len(min_val)]).all():
                break

            # Otherwise remove as much right hand side elements as possible
            i = len(min_val) - 1  # Index to be removed
            while True:
                # If element is safe to be removed, continue
                if min_val[i] > self.max_dist[i]:
                    i -= 1
                # If not slice the distance array and break from the loop
                else:
                    dist_arr = dist_arr[:i + 1, :i + 1]
                    break

            # Reflect from walls and other effects
            for item in self.items:
                item.reflect()  # Reflect particle from walls


class Particle:
    """Class describing particle"""
    rho = 1.0  # Density

    def __init__(self, x, y, vx, vy, radius, color=(200, 0, 255)):
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
