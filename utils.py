from maze import Maze, State, StateType
from mdp import Action, ActionType, MDP
import numpy as np
import pygame
import time

def moving_average(data, window_size=10):
    """Used for smoothing the Training and Validation progress plots"""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def visualize_iteration(maze: Maze,
                        mdp: MDP,
                        n: int,
                        wait: float,
                        timed_visualization: bool = False,
                        iteration_mode: str = 'value',
                        comparing: bool = False,
                        epsilon: float = 1e-4):
    pygame.init()
    screen: pygame.Surface = pygame.display.set_mode(
            (maze.width * maze.cell_size, maze.height * maze.cell_size))
    font = pygame.font.SysFont(None, 32)
    small_font = pygame.font.SysFont(None, 28)
    pygame.display.set_caption(f'iteration mode: {iteration_mode}, Maze: {maze.__repr__()}')

    iteration = 0
    previous_values = {state: 0. for state in mdp.states}

    running = True
    while running:
        if not timed_visualization:
            loop = True
            while loop:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_n:
                            loop = False
                        elif event.key == pygame.K_q:
                            pygame.quit()
                            return
        else:
            time.sleep(wait)

        if iteration_mode == 'value':
            delta = mdp.value_iteration_step()
        elif iteration_mode == 'policy':
            if iteration % n == 0:
                mdp.policy_evaluation()
                policy_stable = mdp.policy_improvement()
                if policy_stable:
                    break
                delta = 0
        else:
            raise ValueError(f"Iteration mode must be either 'value' or 'policy': Iteration_mode: {iteration_mode} is Invalid!")

        iteration += 1

        if iteration % n == 0 or iteration_mode == 'policy':

            screen.fill((255, 255, 255))
            for y in range(maze.height):
                for x in range(maze.width):
                    state = maze.maze[y][x]
                    rect = pygame.Rect(x * maze.cell_size, y * maze.cell_size, maze.cell_size, maze.cell_size)
                    pygame.draw.rect(screen, state.color, rect)
                    pygame.draw.rect(screen, maze.colors['border'], rect, 1)

                    if state.state_type != StateType.WALL:
                        value_change = mdp.state_value_function[state] - previous_values[state]
                        color = (0, 0, 0) if abs(value_change) < 0.1 else (255, 0, 0)

                        if state.state_type == StateType.GOAL:
                            displayed_value = 1.
                        elif state.state_type == StateType.FIRE:
                            displayed_value = -1.
                        else:
                            displayed_value = mdp.state_value_function[state]

                        value_text = font.render(f"{displayed_value:.2f}", True, color)
                        screen.blit(value_text, (x * maze.cell_size + 2, y * maze.cell_size + 2))
                        if state.state_type == StateType.EMPTY:
                            direction = mdp.policy[state]
                            if direction is not None:
                                direction = mdp.policy[state].action_type.value
                                direction_text = font.render(str(direction), True, color)
                                screen.blit(direction_text, (x * maze.cell_size + maze.cell_size - 20, 
                                                            y * maze.cell_size - 20))
            iteration_text = small_font.render(f'Iteration: {iteration}', True, (123, 104, 238))
            screen.blit(iteration_text, (10, maze.screen_height - 20))
            pygame.display.flip()

            previous_values = mdp.state_value_function.copy()

            if iteration_mode == 'value' and delta < epsilon:
                break

    pygame.display.set_caption(f'Iteration_mode: {iteration_mode} | ** Done **')

    while not comparing:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                pygame.quit()
                return