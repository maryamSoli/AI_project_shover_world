import pygame
import sys


def render_env(env):

    if env.render_mode != "human":
        return

    if not hasattr(env, "pygame_initialized") or not env.pygame_initialized:
        pygame.init()
        env.pygame_initialized = True
        window_size = 900
        env.cell_size = window_size // max(env.n_rows, env.n_cols)
        env.window = pygame.display.set_mode((env.cell_size * env.n_cols, env.cell_size * env.n_rows + 120))


    for event in pygame.event.get():

        # Quit window
        if event.type == pygame.QUIT: # Quit the program if the close button is pressed
            pygame.quit()
            sys.exit()

        # Mouse click: move agent to clicked cell
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Convert mouse click to grid coordinates
            mx, my = event.pos # Get mouse coordinates
            r, c = my // env.cell_size, mx // env.cell_size  # Convert to grid coordinates
             # Update agent position if click is inside the grid
            if 0 <= r < env.n_rows and 0 <= c < env.n_cols:
                env.agent_pos = (r, c) # Update agent position

       
        
        elif event.type == pygame.KEYDOWN:
            # Q / ESC → Exit
            if event.key in (pygame.K_q, pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()

            # R → Reset environment
            if event.key == pygame.K_r:
                env.reset()
                continue

            # B → Barrier action
            if event.key == pygame.K_b:
                a = (env.agent_pos[0], env.agent_pos[1], 4)
                env.step(a)
                continue

            # H → Hellify action
            if event.key == pygame.K_h:
                a = (env.agent_pos[0], env.agent_pos[1], 5)
                env.step(a)
                continue

                        # H → Hellify action
            if event.key == pygame.K_n:
                a = (env.agent_pos[0], env.agent_pos[1], 6)
                env.step(a)
                continue
        
            # Arrow keys / WASD movement mapping
            dir_map = {
                pygame.K_UP: 0, pygame.K_w: 0,
                pygame.K_RIGHT: 1, pygame.K_d: 1,
                pygame.K_DOWN: 2, pygame.K_s: 2,
                pygame.K_LEFT: 3, pygame.K_a: 3
            }

            # Perform push action
            if event.key in dir_map:
                act_type = dir_map[event.key]
                a = (env.agent_pos[0], env.agent_pos[1], act_type)
                env.step(a)
                continue
        
       

    # Background
    env.window.fill((245, 245, 245))

    # Draw map tiles
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            val = env.map[r, c] # Get tile type at current cell
            color = (255, 255, 255)
            if val == env.TILE_LAVA:
                color = (230, 60, 60)
            elif val == env.TILE_EMPTY:
                color = (250, 250, 250)
            elif val == env.TILE_BOX:
                color = (255, 215, 0)
            elif val == env.TILE_BARRIER_MARK:
                color = (30, 30, 30)
             # Draw the rectangle for this cell 
            pygame.draw.rect(env.window, color,
                             (c * env.cell_size, r * env.cell_size, env.cell_size, env.cell_size))

    # vertical Grid lines
    for x in range(0, env.n_cols * env.cell_size, env.cell_size):
        pygame.draw.line(env.window, (200, 200, 200), (x, 0), (x, env.n_rows * env.cell_size))
    # horizontal Grid lines
    for y in range(0, env.n_rows * env.cell_size, env.cell_size):
        pygame.draw.line(env.window, (200, 200, 200), (0, y), (env.n_cols * env.cell_size, y))

    # Draw agent cursor as a semi-transparent blue circle
    ar, ac = env.agent_pos    # Agent row and column
    center_x = ac * env.cell_size + env.cell_size // 2    # X center of cell
    center_y = ar * env.cell_size + env.cell_size // 2    # Y center of cell
    radius = max(8, env.cell_size // 4)   # Circle radius (min 8)
    s = pygame.Surface((env.cell_size, env.cell_size), pygame.SRCALPHA)    # Transparent surface
    pygame.draw.circle(s, (30, 144, 255, 160), (env.cell_size // 2, env.cell_size // 2), radius)  # Draw circle
    env.window.blit(s, (ac * env.cell_size, ar * env.cell_size))   # Draw onto main window

    # HUD (text info)
    font = pygame.font.SysFont("Arial", 18)
    info_y = env.n_rows * env.cell_size + 6 # Y coordinate for info text
    env.window.blit(font.render(f"Step: {env.time_step}", True, (10, 10, 10)), (5, info_y)) # Display step number
    env.window.blit(font.render(f"Stamina: {int(env.stamina)}", True, (10, 10, 10)), (110, info_y)) # Display stamina

    last_action = env.last_action if env.last_action is not None else 0
    dir_names = {0: "Up", 1: "Right", 2: "Down", 3: "Left", 4: "Barrier", 5: "Hellify"}
    env.window.blit(font.render(f"Last action: {dir_names.get(int(last_action), str(last_action))}", True, (10, 10, 10)), (250, info_y))
    env.window.blit(font.render(f"Cell: {env.last_pos}", True, (10, 10, 10)), (450, info_y)) # Display last selected cell

    small = pygame.font.SysFont("Arial", 14)  # Display instructions below main info
    instr_y = info_y + 28 #meaning the instruction text will be drawn underneath the info text
    env.window.blit(small.render("Arrows/WASD: push | B: Barrier | H: Hellify | R: Reset | Q: Quit", True, (60, 60, 60)), (5, instr_y))

    # Game over message
    if env.game_over_message:
        env.window.blit(font.render(env.game_over_message, True, (200, 40, 40)), (5, instr_y + 26))  #5  → The x-coordinate 

    pygame.display.flip()  # Update the display with all changes
    pygame.event.pump() # Process internal pygame events
    pygame.time.Clock().tick(1)
