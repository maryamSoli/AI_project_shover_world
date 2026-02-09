import time
import numpy as np
from shover_world.environment import ShoverWorldEnv
from shover_world.gui import render_env

if __name__ == "__main__":
    env = ShoverWorldEnv(
        render_mode='human',
        #n_rows=7, #for hellify
        n_rows=6, # for map2 and barrier
        n_cols=9,
        initial_force=4.0,
        unit_force=1.0,
        map_path=r"D:\AI\1\shover_world\maps\map2.txt",
        #map_path=r"D:\AI\1\shover_world\maps\barrier_maker.txt",
        #map_path=r"D:\AI\1\shover_world\maps\helify.txt",
        seed=0
    )

    obs = env.reset()
    time.sleep(1)
    obs, _, _, _ = env.step((0, 0, 0))  # no-op action to force first GUI update
    render_env(env)
    time.sleep(1)
    done = False
    total_r = 0.0

    print("Initial number of boxes:", int((obs['grid'] == env.TILE_BOX).sum()))

   

    while not done:
        grid = obs['grid']

        # get perfect squares
        squares3 = env._get_perfect_squares(min_size=3)
        squares2 = env._get_perfect_squares(min_size=2)

        # Find coordinates of all individual boxes in the grid.
        # np.where returns two arrays (rows, cols), and zip(*) pairs them into (row, col) tuples.
        box_positions = list(zip(*np.where(grid == env.TILE_BOX)))

        if squares3:  # Hellify available
            r0, c0, s, age = squares3[0]
            action = (r0, c0, 5)
        elif squares2:  # Barrier Maker available
            r0, c0, s, age = squares2[0]
            action = (r0, c0, 4)
        elif box_positions:
            row, col = box_positions[env.random.randint(0, len(box_positions) - 1)]
            act_type = env.random.randint(0, 3) 
            action = (row, col, act_type)
        else:
            # If absolutely no boxes exist on the grid, fall back to a random environment action.
            action = env.action_space.sample()

        obs, reward, done, info = env.step(action)
        total_r += reward

        print(
            f"Step: {info['timestep']}, "
            f"Stamina: {info['stamina']:.1f}, "
            f"Boxes: {info['number_of_boxes']}, "
            f"Destroyed: {info['number_destroyed']}, "
            f"LastValid: {info['last_action_valid']}"
        )

        render_env(env)
        time.sleep(0.01)

    print("Episode return:", total_r)
    env.close()
