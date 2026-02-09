# tests/test_perfect_squares.py
import numpy as np
from shover_world.environment import ShoverWorldEnv


def env_with_map(txt, n=5):
    path = "temp_map.txt"
    with open(path, "w") as f:
        f.write(txt)
    return ShoverWorldEnv(render_mode=None, n_rows=n, n_cols=n, map_path=path)


def test_perfect_square_2x2_detection():
    env = env_with_map("""
0 0 0 0 0
0 1 1 0 0
0 1 1 0 0
0 0 0 0 0
0 0 0 0 0
""")
    env.reset()

    squares = env._get_perfect_squares(min_size=2)
    assert len(squares) == 1
    r, c, size, age = squares[0]
    assert size == 2
    assert (r, c) == (1, 1)


def test_perfect_square_3x3_detection():
    env = env_with_map("""
0 0 0 0 0
0 1 1 1 0
0 1 1 1 0
0 1 1 1 0
0 0 0 0 0
""")
    env.reset()

    squares = env._get_perfect_squares(min_size=3)
    assert len(squares) == 1
    assert squares[0][2] == 3  #size


def test_square_adjacent_exclusion():
    """
    A 2x2 block touching another box adjacent must not be considered perfect.
    """
    env = env_with_map("""
0 0 0 0 0
0 1 1 0 0
0 1 1 1 0
0 0 0 0 0
0 0 0 0 0
""")
    env.reset()

    squares = env._get_perfect_squares(min_size=2)
    assert squares == [] 
