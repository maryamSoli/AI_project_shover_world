# tests/test_stationary.py
import pytest
import numpy as np
from shover_world.environment import ShoverWorldEnv


def env_with_map(txt):
    path = "temp_map.txt"
    with open(path, "w") as f:
        f.write(txt)
    return ShoverWorldEnv(
        render_mode=None, n_rows=4, n_cols=4,
        map_path=path
    )


def test_stationary_updates_across_timesteps():
    """
    stationary should be rebuilt only when map changes; pushing modifies map.
    """
    env = env_with_map("""
0 0 0 0
0 1 0 0
0 1 0 0
0 0 0 0
""")
    env.reset()

    
    assert (1, 1) in env.stationary
    assert (2, 1) in env.stationary

    
    env.step((1, 1, 2))


    st = env.stationary
   
    assert (2, 1) in st
    assert (3, 1) in st
    assert (1, 1) not in st
