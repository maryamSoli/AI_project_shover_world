# ShoverWorld
ShoverWorld is a custom grid-based reinforcement learning environment built using Gymnasium. Agents interact with boxes, lava, barriers, and special “perfect square” structures, practicing spatial reasoning, resource management, and sequential decision-making.

## Installation
Open cmd in windows and switch to the directory
**Basic usage:**
```bash
...\1\shover_world>pip install gymnasium numpy pygame pytest
```
## Running the Environmen
**Basic usage:**
```bash
...\1>python -m shover_world.test 
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_rows` | Number of rows in the grid | 6 |
| `n_cols` | Number of columns in the grid | 6 |
| `max_timestep` | Maximum timesteps per episode | 400 |
| `initial_stamina` | Agent's starting stamina | 1000.0 |
| `initial_force` | Base force applied when pushing boxes | 40.0 |
| `unit_force` | Stamina cost per basic action | 10.0 |
| `perf_sq_initial_age` | Lifespan of perfect-square transformations | 10 |
| `map_path` | Optional path to a predefined map | None |
| `seed` | Optional RNG seed for reproducibility | None |

## Environment Overview

### Grid Tiles
- `0` – Empty  
- `1` – Box  
- `-100` – Lava  
- `100` – Barrier (from perfect-square transformations)  

### Observations
Returns a dictionary containing:  
- `grid` – Current grid state  
- `agent` – Agent position  
- `stamina` – Remaining stamina  
- `previous_selected_position` – Coordinates of the last action target  
- `previous_action` – Last action type  

### Actions
A tuple `(row, col, action_type)` with the following `action_type` options:  
- `0–3` – Move/push in a cardinal direction (up, right, down, left)  
- `4` – Convert a perfect square of boxes into barriers (“Barrier Maker”)  
- `5` – “Hellify” a perfect square: clears outer boxes and fills inner area with lava  

## Map Loading

- Loads maps from a text file.  
- Validates row/column counts and box placement.  
- Initializes stationary box states and perfect square tracking. 

## Mechanics & Features

### Box Movement & Push Chains
- Boxes can be pushed in a chain if aligned.  
- Moving boxes costs stamina proportional to the chain length.  
- Pushing boxes into lava destroys them and increases the destroyed count.  

### Perfect Squares
- The environment detects isolated square blocks of boxes.  
- Adjacent boxes invalidate detection.  
- Perfect squares can be used to create barriers or lava via actions `4` & `5`.  

### Stamina & Rewards
- Invalid moves or illegal pushes reduce stamina.  
- Barrier creation rewards stamina proportional to the square area.  

### Stationary Tracking
- Each box tracks its stationary state per direction to prevent inconsistent movements.  

### Episode Termination
- Stamina reaches `0` → Lose  
- Maximum timestep reached → Episode ends  
- No boxes remaining → Win  


## Tests

- **test_invalid_actions.py** – Invalid action handling & stamina penalties  
- **test_map_loading.py** – Map validation & parsing  
- **test_perfect_square.py** – Detection of perfect squares  
- **test_push_chain.py** – Multi-box push chain mechanics  
- **test_stamina.py** – Stamina updates, including lava interactions  
- **test_stationary.py** – Stationary box tracking and directional state  

**Basic usage:**
```bash
...\1\shover_world>python -m pytest -q
```


