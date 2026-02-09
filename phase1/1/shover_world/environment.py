
# environment.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Optional, Tuple, List, Dict


class ShoverWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self,
                 render_mode: Optional[str] = "human",
                 n_rows: int = 6, n_cols: int = 6,
                 max_timestep: int = 400,
                 initial_stamina: float = 1000.0,
                 initial_force: float = 40.0,
                 unit_force: float = 10.0,
                 perf_sq_initial_age: int = 15, # it will be halfed
                 map_path: Optional[str] = None,
                 seed: Optional[int] = None):

        super().__init__()
        self.n_rows, self.n_cols = n_rows, n_cols
        self.max_timestep = max_timestep
        self.initial_stamina = initial_stamina
        self.initial_force = initial_force
        self.unit_force = unit_force
        self.perf_sq_initial_age = perf_sq_initial_age
        self.map_path = map_path
        self.seed = seed
        self.random = random.Random(seed)

        
        self.TILE_LAVA = -100
        self.TILE_EMPTY = 0
        self.TILE_BOX = 1
        self.TILE_BARRIER_MARK = 100

        
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=-100, high=100, shape=(n_rows, n_cols), dtype=int),
            "agent": spaces.Box(low=0, high=max(n_rows, n_cols) - 1, shape=(2,), dtype=int),
            "stamina": spaces.Box(low=0, high=np.inf, shape=(), dtype=float),
            "previous_selected_position": spaces.Box(low=-1, high=max(n_rows, n_cols) - 1, shape=(2,), dtype=int),
            "previous_action": spaces.Discrete(7)
        })

        
        self.action_space = spaces.Tuple((spaces.Discrete(n_rows),
                                          spaces.Discrete(n_cols),
                                          spaces.Discrete(7)))

        self._init_state()
        self.render_mode = render_mode

        if map_path:
            self.load_map(map_path)

    def _init_state(self):
        # map initialization(If map doesn't exist yet, create an empty map)
        if not hasattr(self, "map") or self.map is None:
            self.map = np.zeros((self.n_rows, self.n_cols), dtype=int)

        # default agent position (1,1) 
        if not hasattr(self, "agent_pos") or self.agent_pos is None:
            self.agent_pos = (1, 1)

        self.stamina = self.initial_stamina
        self.time_step = 0
        self.last_action = None
        self.last_pos = None
        self.last_dir = None

        # perfect squares: list of tuples (r, c, size, age)
        self.perf_squares: List[Tuple[int, int, int, int]] = []

        # position : action
        self.stationary: Dict[Tuple[int, int], List[bool]] = {}

    
        self.destroyed_count = 0
        self.game_over_message = ""

  
    def load_map(self, path: str):
        self._load_map_from_file(path)
        self._rebuild_stationary()
        self._sync_perf_squares_initial()

    def _load_map_from_file(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            ## Read all lines, strip trailing newlines, and ignore empty lines
            lines = [ln.rstrip("\n") for ln in f.readlines() if ln.strip() != ""]
        if len(lines) != self.n_rows:
            raise ValueError(f"Map rows ({len(lines)}) != n_rows ({self.n_rows})")
        
        self.map.fill(self.TILE_EMPTY)
        for r, line in enumerate(lines):
            ## Split the line into individual tile tokens
            tokens = line.strip().split()
            ## Validate that each row has the correct number of columns
            if len(tokens) != self.n_cols:
                raise ValueError(f"Row {r} length {len(tokens)} != n_cols {self.n_cols}")
            
            # Iterate over each column index and token
            for c, tok in enumerate(tokens):
                try:
                    # Convert the token from string to integer (tile value)
                    val = int(tok)
                except Exception as e:
                    raise ValueError(f"Invalid token '{tok}' at row {r}, col {c}") from e
                
                if val == self.TILE_BOX and (r == 0 or c == 0 or r == self.n_rows - 1 or c == self.n_cols - 1):
                    if self.n_rows <= 2 or self.n_cols <= 2:
                        raise ValueError(f"Box on edge at {r},{c}")
                
                # # Assign the parsed value to the map at row r, column c  
                self.map[r, c] = val

   
    def _find_perfect_squares_raw(self, min_size: int = 2) -> List[Tuple[int, int, int]]: #(row, col, size)
        squares = []
        for size in range(min_size, min(self.n_rows, self.n_cols) + 1):
            ## Iterate over all possible top-left corners for the square
            for r in range(0, self.n_rows - size + 1):
                # Iterate over all possible top-left column positions for the square
                for c in range(0, self.n_cols - size + 1):
                    #   Extract the subgrid (block) corresponding to the current square
                    block = self.map[r:r + size, c:c + size]
                    #   Check if full block is boxes
                    if np.all(block == self.TILE_BOX):
                        top = max(r - 1, 0)
                        left = max(c - 1, 0)
                        bottom = min(r + size, self.n_rows - 1)
                        right = min(c + size, self.n_cols - 1)
                         # Look for adjacent boxes external to the square
                        adj = self.map[top:bottom + 1, left:right + 1].copy()
                        # Mask the inner square so only the surrounding adjacency remains
                        inner_r0 = r - top
                        inner_c0 = c - left
                        adj[inner_r0:inner_r0 + size, inner_c0:inner_c0 + size] = 0

                        if not np.any(adj == self.TILE_BOX):
                            squares.append((r, c, size))
        return squares

    #Detects all initial perfect squares and resets their ages to 0.
    def _sync_perf_squares_initial(self):
        raw = self._find_perfect_squares_raw(min_size=2)
        self.perf_squares = [(r, c, s, 0) for (r, c, s) in raw]

    def _update_perf_squares(self):
        # Convert to a set to remove duplicates
        raw_set = set(self._find_perfect_squares_raw(min_size=2))
        #mapping each square (row, col, size) to its previous age
        prev_map = {(r, c, s): age for (r, c, s, age) in self.perf_squares}
        new_list = []
        for (r, c, s) in raw_set:
            key = (r, c, s)
            age = prev_map.get(key, 0) + 1 # Increment age: use previous age otherwise start at 0

            # If the square's age reaches or exceeds the initial age threshold, remove it
            if age >= self.perf_sq_initial_age:
                self.map[r:r + s, c:c + s] = self.TILE_EMPTY # Clear the square from the map
            else:
                new_list.append((r, c, s, age))
        self.perf_squares = new_list

    def _get_perfect_squares(self, min_size: int = 2):
        return [sq for sq in self.perf_squares if sq[2] >= min_size]

    def _rebuild_stationary(self):
       
        self.stationary.clear()
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self.map[r, c] == self.TILE_BOX:
                    self.stationary[(r, c)] = [True] * 4

    def _longest_chain_in_direction(self, act_type: int) -> int:
       
        if act_type in (1, 3):  #horizontal
            maxlen = 0 # maximum contiguous boxes in a row
            for r in range(self.n_rows):
                current = 0 # current contiguous box count in this row
                for c in range(self.n_cols):
                    if self.map[r, c] == self.TILE_BOX:
                        current += 1 # increment current chain length
                        maxlen = max(maxlen, current)  # update max if current longer
                    else:
                        current = 0  # reset chain if no box
            return maxlen #  longest horizontal chain found
        
        elif act_type in (0, 2):  
            maxlen = 0
            for c in range(self.n_cols):
                current = 0
                for r in range(self.n_rows):
                    if self.map[r, c] == self.TILE_BOX:
                        current += 1
                        maxlen = max(maxlen, current)
                    else:
                        current = 0
            return maxlen
        return 0

    
    def step(self, action: Tuple[int, int, int]):
       
        row, col, act_type = action
        self.time_step += 1
        reward = 0.0
        lava_destroyed = 0
        last_action_valid = True
        self.last_action = int(act_type)
        self.last_pos = (row, col)

        
        push_chain_length = 0
        push_blocked = False
        initial_force_charged = False

        directions = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        self.last_dir = act_type if act_type < 4 else None

       # Barrier Maker action (act_type=4)
        if act_type == 4:
            squares = self._get_perfect_squares(min_size=2)
            if squares:
                
                r0, c0, size, age = squares[0]
                
                self.map[r0:r0 + size, c0:c0 + size] = self.TILE_BARRIER_MARK
               
                self.stamina += size * size
                reward += float(size * size)
                last_action_valid = True

                # Update perfect squares and stationary boxes
                self._sync_perf_squares_initial()
               
                self._rebuild_stationary()
            else:
                last_action_valid = False
                self.stamina -= self.unit_force

        # Hellify action (act_type=5)
        elif act_type == 5:
            squares = self._get_perfect_squares(min_size=3)
            if squares:
                r0, c0, size, age = squares[0]
                
                # Remove outer boxes
                self.map[r0, c0:c0 + size] = self.TILE_EMPTY
                self.map[r0 + size - 1, c0:c0 + size] = self.TILE_EMPTY
                self.map[r0:r0 + size, c0] = self.TILE_EMPTY
                self.map[r0:r0 + size, c0 + size - 1] = self.TILE_EMPTY
                
                # Replace inner area with lava if size>2
                if size > 2:
                    self.map[r0 + 1:r0 + size - 1, c0 + 1:c0 + size - 1] = self.TILE_LAVA
               
                self.destroyed_count += size ** 2
                last_action_valid = True
                
                self._sync_perf_squares_initial()
                
                self._rebuild_stationary()
            else:
                last_action_valid = False
                self.stamina -= self.unit_force

        elif act_type == 6:

            r, c = self.agent_pos
           
            r_up = r - 1
            if 0 <= r_up < self.n_rows and self.map[r_up, c] == self.TILE_EMPTY:
                self.map[r_up, c] = self.TILE_BARRIER_MARK  
                reward = 1.0
                last_action_valid = True
                self._rebuild_stationary()
            else:
                last_action_valid = False
                self.stamina -= self.unit_force

            
            self.map[r, c] = self.TILE_BOX


        elif act_type in directions:
            #dr, dc = direction vector for chosen push action
            dr, dc = directions[act_type]

           
            if not (0 <= row < self.n_rows and 0 <= col < self.n_cols):
                last_action_valid = False
                self.stamina -= self.unit_force

           
            elif self.map[row, col] != self.TILE_BOX:
                 # How many boxes would be in front 
                push_chain_length = self._longest_chain_in_direction(act_type)
                last_action_valid = False
                self.stamina -= self.unit_force

            else:
                
                boxes_chain = []
                r0, c0 = row, col
                while 0 <= r0 < self.n_rows and 0 <= c0 < self.n_cols and self.map[r0, c0] == self.TILE_BOX:
                    boxes_chain.append((r0, c0))
                    r0 += dr
                    c0 += dc

                k = len(boxes_chain)
                push_chain_length = k

                #[(2,1), (2,2), (2,3)]
                front_pos = (boxes_chain[0][0], boxes_chain[0][1])

                # Look up whether this first box is stationary in all directions.
                front_stationary_flags = self.stationary.get(front_pos, [True] * 4)

                #Check if the box is stationary in the direction you are pushing
                if front_stationary_flags[act_type]:
                    
                    self.stamina -= self.initial_force
                    reward -= float(self.initial_force) 
                    initial_force_charged = True

              
                self.stamina -= self.unit_force * max(k, 1)

                moved = False
                moved_positions = []  
                for r0, c0 in reversed(boxes_chain):
                    nr, nc = r0 + dr, c0 + dc
                    if not (0 <= nr < self.n_rows and 0 <= nc < self.n_cols):
                        break

                    if self.map[nr, nc] == self.TILE_EMPTY:
                        self.map[nr, nc] = self.TILE_BOX
                        self.map[r0, c0] = self.TILE_EMPTY
                        moved = True
                        moved_positions.append((nr, nc))
                    elif self.map[nr, nc] == self.TILE_LAVA:
                        # Box is removed
                        self.map[r0, c0] = self.TILE_EMPTY
                        lava_destroyed += 1
                        self.destroyed_count += 1
                        moved = True
                        
                        self.stamina += self.initial_force
                        reward += float(self.initial_force) 
                    else:
                       
                        pass

                last_action_valid = moved

                
                push_blocked = (push_chain_length > 0 and not moved)

               
                self._update_perf_squares()
                self._rebuild_stationary()
                for (mr, mc) in moved_positions:
                 
                    if (mr, mc) in self.stationary:
                        self.stationary[(mr, mc)][act_type] = False

        else:
            
            last_action_valid = False
            self.stamina -= self.unit_force

      
        self.stamina = max(0.0, self.stamina)

        
        self._update_perf_squares()

       
        done = False
        if self.stamina <= 0:
            done = True
            self.game_over_message = "Lose (Stamina 0)"
        elif self.time_step >= self.max_timestep:
            done = True
            self.game_over_message = "Max Timestep"
        elif not np.any(self.map == self.TILE_BOX):
            done = True
            self.game_over_message = "Win (All boxes cleared)"

        return self._get_obs(), reward, done, self._get_info(
            last_action_valid=last_action_valid,
            lava_destroyed_this_step=lava_destroyed,
            push_chain_length=push_chain_length,
            push_blocked=push_blocked,
            initial_force_charged=initial_force_charged
        )

    def _get_obs(self):
        return {
            "grid": self.map.copy(),
            "agent": np.array(self.agent_pos),
            "stamina": float(self.stamina),
            "previous_selected_position": np.array(self.last_pos if self.last_pos else [-1, -1]),
            "previous_action": int(self.last_action) if self.last_action is not None else 0
        }

    def _get_info(self, **kwargs):
        return {
            "timestep": self.time_step,
            "stamina": self.stamina,
            "number_of_boxes": int(np.sum(self.map == self.TILE_BOX)),
            "number_destroyed": self.destroyed_count,
            "last_action_valid": kwargs.get("last_action_valid", True),
            "lava_destroyed_this_step": kwargs.get("lava_destroyed_this_step", 0),
            "perfect_squares_available": [(r, c, s) for r, c, s, age in self.perf_squares],
            "push_chain_length": kwargs.get("push_chain_length", 0),
            "push_blocked": kwargs.get("push_blocked", False),
            "initial_force_charged": kwargs.get("initial_force_charged", False),
        }

    def reset(self):
        self.time_step = 0
        self._init_state()
        if self.map_path:
            self.load_map(self.map_path)
        else:
            raise ValueError("No map_path provided; cannot generate map")
        return self._get_obs()

    def render(self):
        return

    def close(self):
        return
