# Random Maze Generator using Depth-first Search
# http://internal_representation_analysis.activestate.com/recipes/578356-random-maze-generator/
import random
from envs.grid_2d_env.Cell import Cell

def generate_maze(width, height):
    maze = [[1 for x in range(width)] for y in range(height)]
    dx = [0, 1, 0, -1]; dy = [-1, 0, 1, 0] # 4 directions to move in the maze
    # start the maze from a random cell
    stack = [(random.randint(0, width - 1), random.randint(0, height - 1))]

    while len(stack) > 0:
        (cx, cy) = stack[-1]
        maze[cy][cx] = 0
        # find a new cell to add
        nlst = [] # list of available neighbors
        for i in range(4):
            nx = cx + dx[i]; ny = cy + dy[i]
            if nx >= 0 and nx < width and ny >= 0 and ny < height:
                if maze[ny][nx] == 1:
                    # of occupied neighbors must be 0
                    ctr = 0
                    for j in range(4):
                        ex = nx + dx[j]; ey = ny + dy[j]
                        if ex >= 0 and ex < width and ey >= 0 and ey < height:
                            if maze[ey][ex] == 0: ctr += 1
                    if ctr == 1: nlst.append(i)
        # if 1 or more neighbors available then randomly select one and move
        if len(nlst) > 0:
            ir = nlst[random.randint(0, len(nlst) - 1)]
            cx += dx[ir]; cy += dy[ir]
            stack.append((cx, cy))
        else: stack.pop()

    return maze

def generate_random_maze_cells(width, height):
    cell_values = generate_maze(width, height)
    cells = []
    for r in range(height):
        row = []
        for c in range(width):
            can_step = True
            is_transparent = True
            if cell_values[r][c] == 1:
                can_step = False
                is_transparent = False

            row.append(Cell(cell_values[r][c], can_step, is_transparent))
        cells.append(row)
    return cells