"""common.py"""

import numpy as np
import pygame

# Colors
BLACK = pygame.Color((0, 0, 0))
WHITE = pygame.Color((255, 255, 255))
BLUE = pygame.Color((0, 128, 255))
RED = pygame.Color((255, 0, 0))
PURPLE = pygame.Color((200, 0, 255))


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
