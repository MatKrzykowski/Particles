"""config.py"""

from dataclasses import dataclass

@dataclass()
class Config():
    width: int
    height: int
    fontsize: int

default_config = Config(
    800,  # window width
    600,  # window height
    18,  # fontsize
)