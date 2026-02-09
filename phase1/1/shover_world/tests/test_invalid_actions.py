# tests/test_invalid_actions.py
import numpy as np
import pytest
from shover_world.environment import ShoverWorldEnv


def env_with_map(txt):
    path = "temp_map.txt"
    with open(path, "w") as f:
        f.write(txt)
    return ShoverWorldEnv(render_mode=None, n_rows=5, n_cols=5, map_path=path)


def test_invalid_barrier_action():
    """
    No perfect square available → invalid.
    """
    env = env_with_map("""
0 0 0 0 0
0 1 0 0 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
""")
    env.reset()

    obs, reward, done, info = env.step((1, 1, 4))  # barrier
    assert info["last_action_valid"] is False


def test_invalid_hellify_action():
    env = env_with_map("""
0 0 0 0 0
0 1 1 0 0
0 1 1 0 0
0 0 0 0 0
0 0 0 0 0
""")
    env.reset()

    obs, reward, done, info = env.step((1, 1, 5))  # hellify but only 2x2
    assert info["last_action_valid"] is False


def test_push_into_lava_removes_box():
    env = env_with_map("""
0 0 0 0 0
0 1 0 0 0
0 -100 0 0 0
0 0 0 0 0
0 0 0 0 0
""")
    env.reset()

    obs, reward, done, info = env.step((1, 1, 2))  # push down into lava

    # This assertion checks that the environment correctly left lava
    # at position (1,1), meaning the box was destroyed.
    assert obs["grid"][2, 1] == env.TILE_LAVA


def test_push_into_barrier_invalid():
    env = env_with_map("""
0 0 0 0 0
0 1 100 0 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
""")
    env.reset()

    obs, reward, done, info = env.step((1, 1, 1))

    assert info["last_action_valid"] is False
