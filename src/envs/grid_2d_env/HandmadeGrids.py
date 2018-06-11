from .Cell import Cell
from .PositionAndRotation import PositionAndRotation


def make_topological_setup(eval=False):
    topo = []
    for r in range(5):
        row = []
        for c in range(9):
            row.append(Cell(1,False,False))
        topo.append(row)
    non_walls = [(0,1), (0,7), (1,0),(1,1),(1,2),(1, 3),(1, 4),
                 (1, 5),(1,6),(1,7),(2,1),(2,7),(2,8),(3,1),
                 (3,7),(4, 1),(4, 2),(4, 3),(4,4),(4,5),(4,6),(4,7)]
    for (x,y) in non_walls:
        topo[x][y].value = 0
        topo[x][y].can_step = True
        topo[x][y].is_transparent = True
    topo[4][4].value = 20
    topo[4][4].is_transparent = False
    topo[2][8].value = 30
    topo[2][8].is_transparent = False

    #For evaluation
    if eval:
        topo[2][1].value = 1
        topo[2][1].is_transparent = False
        topo[2][1].can_step = False

    # Hardcoded goal images.
    if eval:
        img_goal_id_1 = 75
    else:
        img_goal_id_1 = 79
    img_goal_id_2 = 45

    task_1 = (PositionAndRotation(0,1,0), (4,4), 20, img_goal_id_1)
    task_2 = (PositionAndRotation(1,0,90), (2,8), 30, img_goal_id_2)
    task_3 = (PositionAndRotation(0,7,0), (4,4), 20, img_goal_id_1)
    task_list = [task_1, task_2, task_3]

    return topo, task_list

def make_metric_triangle_setup(eval=False):
    metric = []
    walls = [(x,y) for x in range(1,4) for y in range(1,4)]
    # Hide A and B from each other.
    mist = [(0,1), (0,2), (0,3)]

    for r in range(5):
        row = []
        for c in range(5):
            row.append(Cell(0,True,True))
        metric.append(row)

    for pos in mist:
        metric[pos[0]][pos[1]].is_transparent = False
        if eval:
            metric[pos[0]][pos[1]].can_step = True
            metric[pos[0]][pos[1]].value = 0
            metric[pos[0]][pos[1]].is_transparent = False
        else:
            metric[pos[0]][pos[1]].can_step = False
            metric[pos[0]][pos[1]].value = 1

    for (x,y) in walls:
        metric[x][y].value = 1
        metric[x][y].can_step = False
        metric[x][y].is_transparent = False

    metric[0][0].value = 20
    metric[0][0].can_step = True
    metric[0][0].is_transparent = False

    metric[0][4].value = 30
    metric[0][4].can_step = True
    metric[0][4].is_transparent = False

    metric[4][2].value = 40
    metric[4][2].can_step = True
    metric[4][2].is_transparent = False

    if eval:
        # Hardcoded goal images.
        img_goal_id_A = 22
        img_goal_id_B = 26

        task_1 = (PositionAndRotation(0, 0, 90), (0, 4), 30, img_goal_id_B)
        task_2 = (PositionAndRotation(0, 4, 270), (0, 0), 20, img_goal_id_A)
        task_list = [task_1, task_2]
    else:
        # Hardcoded goal images.
        img_goal_id_A = 22
        img_goal_id_B = 26
        img_goal_id_C = 49

        task_1 = (PositionAndRotation(0,0,90), (4,2), 40, img_goal_id_C)
        task_2 = (PositionAndRotation(4,2,270), (0,0), 20, img_goal_id_A)
        task_3 = (PositionAndRotation(0,4,270), (4,2), 40, img_goal_id_C)
        task_4 = (PositionAndRotation(4,2,90), (0,4), 30, img_goal_id_B)
        task_list = [task_1, task_2, task_3, task_4]

    return metric, task_list

