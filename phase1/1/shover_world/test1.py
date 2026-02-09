import time
import numpy as np
from shover_world.environment import ShoverWorldEnv
from shover_world.gui import render_env


if __name__ == "__main__":

    
    env = ShoverWorldEnv(
        render_mode='human',
        n_rows=7, #for hellify
        #n_rows=6,
        n_cols=9,
        initial_force=4.0,
        unit_force=1.0,
       
        #map_path=r"D:\AI\1\shover_world\maps\map2.txt",
        #map_path=r"D:\AI\1\shover_world\maps\barrier_maker.txt",
        map_path=r"D:\AI\1\shover_world\maps\helify.txt",
        seed=0
    )

    
    obs = env.reset()
    done = False

    print("Game starting in 3 seconds...")
    time.sleep(3)

    
    while not done:

        
        render_env(env)

        
        time.sleep(0.01)

       

    env.close()
    print("Game finished.")
