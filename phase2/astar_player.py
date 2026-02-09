import heapq
import copy
import time
import numpy as np
from dataclasses import dataclass
from environment import ShoverWorldEnv
import pygame
import sys



ACTION_NAMES = {
    0: "PUSH ↑",
    1: "PUSH →",
    2: "PUSH ↓",
    3: "PUSH ←",
    4: "PLACE BARRIER",
    5: "HELLIFY"
}


DIR_OFFSET = {
    0: (-1, 0),   # up
    1: (0, 1),    # right
    2: (1, 0),    # down
    3: (0, -1)    # left
}


# A "search state" used in A*
@dataclass(frozen=True)
class State:
    grid: tuple     # snapshot of the map
    stamina: int    # remaining stamina


def env_to_state(env):
    """
    Convert the environment object into a hashable State.
    numpy arrays cannot be placed in sets, so convert them to tuples.
    """
    return State(
        tuple(map(tuple, env.map)),  # convert grid to nested tuples
        int(env.stamina)
    )


def heuristic(env):
    """
    A* heuristic:
    Larger value means "worse" state, so A* avoids it.

    - Each remaining box increases cost
    - Perfect square blocks slightly reduce cost 
    """
    return (
        np.sum(env.map == env.TILE_BOX) * 10
        - len(env._get_perfect_squares(min_size=2)) * 5
    )


def generate_actions(env):
    """
    Generate ALL possible actions.

    For every box cell -> try pushing it in every direction.

    Also add two "global" actions:
    - place barrier
    - hellify

    
    """
    actions = []
    dirs = [0, 1, 2, 3]  # up, right, down, left

    for r in range(env.n_rows):
        for c in range(env.n_cols):
            # Only consider tiles with boxes
            if env.map[r, c] == env.TILE_BOX:
                for d in dirs:
                    actions.append((r, c, d))

    
    actions.append((0, 0, 4))  # place barrier
    actions.append((0, 0, 5))  # hellify

    return actions


def astar_plan(env):
    # Priority queue (min-heap). Always pops the "best" node first.
    pq = []

    # Set of states we have already visited so we do not repeat work.
    visited = set()

    # Counter used as a tie-breaker inside the heap(if 2 f are equal it comares the counter).
    counter = 0

    # Push the START NODE into the heap.
    # Each heap entry is a tuple:
    # (f, g, counter, state, env_copy, path)
    heapq.heappush(
        pq,
        (
            heuristic(env),        # f = estimated total cost (here: only heuristic)
            0,                     # g = cost so far = 0 (we have not moved yet)
            counter,               
            env_to_state(env),     # logical state 
            copy.deepcopy(env),    # a working copy of the environment
            []                     # path of actions we took so far
        )
    )
    counter += 1   # next pushed node will get a different id

    # MAIN LOOP: runs until heap is empty
    while pq:

        # Pop the node with the smallest f value
        f, g, _, state, cur_env, path = heapq.heappop(pq)

        # If we've already seen this exact state, skip it.
        if state in visited:
            continue

        # Mark state as visited
        visited.add(state)

        # GOAL TEST:
        # If there are no boxes left, solution found. Return the path.
        if not np.any(cur_env.map == cur_env.TILE_BOX):
            return path

        # Otherwise: expand this node (generate possible next actions)
        for action in generate_actions(cur_env):

            # Make a deep copy so we can simulate safely
            next_env = copy.deepcopy(cur_env)

            # Apply the action
            _, _, _, info = next_env.step(action)

            # If the action was illegal → ignore
            if not info["last_action_valid"]:
                continue

            # Convert next environment to logical state
            next_state = env_to_state(next_env)

            # If we already visited this state → ignore
            if next_state in visited:
                continue

            # Compute heuristic on this new state
            h = heuristic(next_env)

            # Push new node to heap:
            # new_f = new_g + heuristic
            heapq.heappush(
                pq,
                (
                    g + 1 + h,          # f
                    g + 1,              # new g cost
                    counter,            # tie-breaker
                    next_state,         # frozen state
                    next_env,           # full environment copy
                    path + [action]     # extend path
                )
            )
            counter += 1

    # Heap empty, no solution
    return None



def render_env(env):
    """
    Handles pygame rendering. Pure UI, no algorithmic effect.
    """

    if env.render_mode != "human":
        return

    # Initialize pygame once
    if not hasattr(env, "pygame_initialized"):
        pygame.init()
        env.pygame_initialized = True

        window_size = 700
        env.cell_size = window_size // max(env.n_rows, env.n_cols)

        env.window = pygame.display.set_mode(
            (env.cell_size * env.n_cols,
             env.cell_size * env.n_rows + 130)
        )

        pygame.display.set_caption("ShoverWorld AI")

    # Background
    env.window.fill((245, 245, 245))

    # Draw grid tiles based on value
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            val = env.map[r, c]
            color = (255, 255, 255)

            if val == env.TILE_LAVA:
                color = (230, 60, 60)
            elif val == env.TILE_BOX:
                color = (255, 215, 0)
            elif val == env.TILE_BARRIER_MARK:
                color = (30, 30, 30)

            pygame.draw.rect(
                env.window,
                color,
                (c * env.cell_size,
                 r * env.cell_size,
                 env.cell_size,
                 env.cell_size),
            )

    # Draw grid lines
    for x in range(0, env.n_cols * env.cell_size, env.cell_size):
        pygame.draw.line(env.window, (200, 200, 200),
                         (x, 0), (x, env.n_rows * env.cell_size))

    for y in range(0, env.n_rows * env.cell_size, env.cell_size):
        pygame.draw.line(env.window, (200, 200, 200),
                         (0, y), (env.n_cols * env.cell_size, y))

    # Status text
    font = pygame.font.SysFont("Arial", 15)
    info_y = env.n_rows * env.cell_size + 5
    lh = 18

    env.window.blit(
        font.render(f"Step: {env.time_step}", True, (0, 0, 0)),
        (5, info_y)
    )

    env.window.blit(
        font.render(f"Stamina: {int(env.stamina)}", True, (0, 0, 0)),
        (120, info_y)
    )

    # If environment returned info about last action, display it
    if env.last_info:
        env.window.blit(
            font.render(
                f"Boxes Left: {env.last_info['number_of_boxes']} | "
                f"Destroyed: {env.last_info['number_destroyed']} | "
                f"Valid: {env.last_info['last_action_valid']}",
                True, (0, 0, 0)
            ),
            (5, info_y + lh)
        )

        env.window.blit(
            font.render(
                f"Action: {env.last_action_name}",
                True, (0, 0, 0)
            ),
            (5, info_y + 2 * lh)
        )

        if env.last_box_before is not None:
            env.window.blit(
                font.render(
                    f"Box: {env.last_box_before} → {env.last_box_after}",
                    True, (0, 0, 0)
                ),
                (5, info_y + 3 * lh)
            )

    # Game over message if exists
    if env.game_over_message:
        env.window.blit(
            font.render(env.game_over_message, True, (200, 40, 40)),
            (420, info_y)
        )

    pygame.display.flip()
    pygame.event.pump()


# Program entry point
if __name__ == "__main__":

    # Create environment
    env = ShoverWorldEnv(
        render_mode="human",
        n_rows=13,
        n_cols=9,
        initial_stamina=1000,
        unit_force=10,
        initial_force=40,
        map_path="challenge_map.txt",
    )

    env.reset()

    # Debug metadata for rendering
    env.last_action_name = ""
    env.last_box_before = None
    env.last_box_after = None
    env.last_info = {}

    print("Planning with A* ...")

    # Compute plan using A*
    plan = astar_plan(env)

    if plan is None:
        print(" No solution found")
        sys.exit(0)

    # Execute plan
    for step in plan:
        r, c, d = step

        env.last_action_name = ACTION_NAMES.get(d, "UNKNOWN")

        # Track which box moved
        if d in DIR_OFFSET:
            dr, dc = DIR_OFFSET[d]
            env.last_box_before = (r, c)
            env.last_box_after = (r + dr, c + dc)
        else:
            env.last_box_before = None
            env.last_box_after = None

        _, _, done, info = env.step(step)
        env.last_info = info

        render_env(env)
        time.sleep(1.5)
        # Stop if episode ends
        if done:
            render_env(env)
            break

    # Keep window open at the end
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
        time.sleep(0.1)
