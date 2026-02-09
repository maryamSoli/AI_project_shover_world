# tests/test_stamina.py
import numpy as np
import pytest
from shover_world.environment import ShoverWorldEnv


def env_with_map(txt):
    path = "temp_map.txt"
    with open(path, "w") as f:
        f.write(txt)
    return ShoverWorldEnv(
        render_mode=None, n_rows=5, n_cols=5,
        map_path=path,
        initial_stamina=100, unit_force=2
    )


def test_baseline_invalid_cost_minus_unit():
    env = env_with_map("""
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
""")
    env.reset()

    
    obs, reward, done, info = env.step((2, 2, 0))
    assert info["stamina"] == pytest.approx(100 - 2)



def test_push_cost_formula():
    env = env_with_map("""
0 0 0 0 0
0 1 1 0 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
""")
    env.reset()

   
    print("ENVIRONMENT PARAMETERS:")
    print(f"initial_stamina: {env.initial_stamina}")
    print(f"initial_force: {env.initial_force}")
    print(f"unit_force: {env.unit_force}")
    
    obs, reward, done, info = env.step((1, 1, 1))
    
   
    expected_stamina = env.initial_stamina - env.initial_force - (env.unit_force * 2)
    assert info["stamina"] == pytest.approx(expected_stamina)

def test_barrier_maker_gain():
    """
    2x2 square gives +4 stamina reward.
    """
    env = env_with_map("""
0 0 0 0 0
0 1 1 0 0
0 1 1 0 0
0 0 0 0 0
0 0 0 0 0
""")
    env.reset()

    
    obs, reward, done, info = env.step((1, 1, 4))

    assert reward == 4
    assert info["stamina"] == pytest.approx(100 + 4)


def test_hellify_cost_and_gain():
    """
    Hellify on 3x3 removes outer shell, replaces inside with lava,
    reward = destroyed boxes counted internally.
    """
    env = env_with_map("""
0 0 0 0 0
0 1 1 1 0
0 1 1 1 0
0 1 1 1 0
0 0 0 0 0
""")
    env.reset()

    obs, reward, done, info = env.step((0, 0, 5))  # hellify

    
    assert info["last_action_valid"] is True

    
    assert info["number_destroyed"] == 9

    
    grid = obs["grid"]

    
    assert grid[2, 2] == env.TILE_LAVA

    
    edge_positions = [
        (1,1), (1,2), (1,3),
        (2,1),        (2,3),
        (3,1), (3,2), (3,3)
    ]
    for r, c in edge_positions:
        if (r, c) != (2, 2):
            assert grid[r, c] == env.TILE_EMPTY

