import numpy as np
import pytest
from shover_world.environment import ShoverWorldEnv


def env_with_map(txt):
    path = "temp_map.txt"
    with open(path, "w") as f:
        f.write(txt)
    return ShoverWorldEnv(
        render_mode=None, n_rows=5, n_cols=5,
        map_path=path, initial_stamina=1000, unit_force=1
    )



def test_push_chain_k_and_blocking():
    env = env_with_map("""
0 0 0 0 0
0 1 1 1 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
""")

    obs = env.reset()
    
    
    print("ENVIRONMENT PARAMETERS:")
    print(f"initial_stamina: {env.initial_stamina}")
    print(f"initial_force: {env.initial_force}")
    print(f"unit_force: {env.unit_force}")
    
    obs, reward, done, info = env.step((1, 1, 1))

    
    expected_stamina = env.initial_stamina - env.initial_force - (env.unit_force * 3)
    assert info["stamina"] == pytest.approx(expected_stamina)


def test_push_blocked_when_barrier():
    """
    If the chain hits a barrier tile, no movement occurs.
    """
    env = env_with_map("""
0 0 0 0 0
0 1 1 100 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
""")

    obs = env.reset()
    obs, reward, done, info = env.step((1, 1, 1)) 

    
    g = obs["grid"]
     # Boxes should NOT have moved because barrier blocks them.
    assert g[1, 1] == env.TILE_BOX
    assert g[1, 2] == env.TILE_BOX

    assert info["last_action_valid"] is False


    # Move agent manually → the agent’s position affects which boxes can be pushed
    env.agent_pos = (2,1)

    # Reset timestep and internal state for a fresh push attempt
    env.time_step = 0
    env._init_state()

  
    obs, reward, done, info = env.step((2,2,1))  

    # Check that the environment reports the push chain length
    #“Make sure the environment returned push_chain_length in its info.
    #If not, stop and error with: Push chain info missing.”
    assert "push_chain_length" in info, "Push chain info missing"
    # The push chain should involve at least 2 boxes
    assert info["push_chain_length"] >= 2
    # Check that the environment reports whether the push was blocked
    # push_blocked is a boolean: True if the push failed, False if successful
    assert isinstance(info["push_blocked"], bool)
