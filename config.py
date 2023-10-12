"""config.py"""

from dataclasses import dataclass

@dataclass()
class Config():
    width: int
    height: int
    subgid_size: int
    fontsize: int
    n_subframes: int
    fps: int

default_config = Config(
    800,  # window width
    600,  # window height
    100,  # subgid size
    18,  # fontsize
    4,  # number of subframes
    60,  # goal fps
)