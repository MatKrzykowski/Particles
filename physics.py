"""physics.py"""

import numpy as np

from particle import Particle


def elastic_collision(i: Particle, j: Particle, dist: float) -> None:
    """Simulates elastic collision between 2 particles, mutating both."""
    i_pos, i_vel = i.position, i.velocity
    j_pos, j_vel = j.position, j.velocity
    pos_diff = i_pos - j_pos
    scalar = np.sum((pos_diff) * (i_vel - j_vel))
    if scalar < 0.0:  # Check if collision didn't already happen
        A = 2 * scalar / (i.mass + j.mass) / dist**2
        i.velocity, j.velocity = \
            i_vel - j.mass * A * pos_diff, \
            j_vel - i.mass * A * -pos_diff
