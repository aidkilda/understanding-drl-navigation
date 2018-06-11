"""
This script is used to visualize and manually test the Grid2DEnvAdapter.
"""

from constants import config

from target_driven_method.Grid2DEnvAdapterForTargetDriven import Grid2DEnvAdapter
import pygame

# Maps keyboard keys to actions indexes in the environment.
key_action_index_map = {
    pygame.K_w: 0,
    pygame.K_s: 1,
    pygame.K_a: 2,
    pygame.K_d: 3
}

if config['task_list']:
    tasks = config['task_list']
    task = tasks[0]
    config['initial_agent'] = task[0]
    config['initial_target'] = task[1]
    config['target_value'] = task[2]
    print("Target id:", task[3])

env = Grid2DEnvAdapter(config)
env.render()

# Renders the given environment and allows navigation in it using keyboard keys
# specified in the action_map above.
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            print("Agent position and rotation before action", env.grid_env.grid_model.agent.position_and_rotation.to_list())
            print("State id before action", env.current_state_id)
            if event.key in key_action_index_map:
                action_index = key_action_index_map[event.key]
                env.step(action_index)
                print("State", env.s_t)
                print("Observation", env.observation)
                print("Agent position and rotation", env.grid_env.grid_model.agent.position_and_rotation.to_list())
                print("Reward: %f, Terminated: %s" % (env.reward, env.terminal))
                print("State id", env.current_state_id)
                print("-----------")
                if env.terminal:
                    env.reset()
                env.render()