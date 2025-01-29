from typing import Dict, List, Tuple
from maze import Maze, StateType
from mdp import Action, ActionType, MDP
from utils import visualize_iteration
import matplotlib.pyplot as plt
import numpy as np
import time

def compare_runtime(maze_configs: List[Tuple[Tuple[int, int], int, int, float]],
                    discount_factors: List[float],
                    iter_: int,
                    noise: float = 0.2,
                    epsilon: float = 1e-4,
                    visualize: bool = False,
                    plot: bool = False,
                    title: str = 'Runtime Comparison of Value Iteration and Policy Iteration'
                    ):

    times: Dict[str, Dict[Tuple[Tuple[int, int], int, int, float], List[float]]] = {'value': {}, 'policy': {}}

    print(f'Average runtime over {iter_} iterations per configuration:')
    print('-'*30)
    for config in maze_configs:
        size, n_goals, n_fires, p = config
        for gamma in discount_factors:
            maze = Maze(width=size[0], height=size[1], p=p, n_goals=n_goals, n_fires=n_fires, screen_width=75, screen_height=55)
            states = maze.maze.flatten()

            actions = {}
            rewards = {}
            for state in states:
                actions[state] = []
                for action_type in [ActionType.LEFT, ActionType.RIGHT, ActionType.UP, ActionType.DOWN]:
                    action = Action(maze, state, action_type, noise=noise)
                    actions[state].append(action)
                    for next_state, _ in action.next_states:
                        if next_state.state_type == StateType.GOAL:
                            rewards[(action.state, action, next_state)] = 1.
                        elif next_state.state_type == StateType.FIRE:
                            rewards[(action.state, action, next_state)] = -1.
                        else:
                            rewards[(action.state, action, next_state)] = 0.0

            mdp = MDP(maze, states, actions, rewards, discount_factor=gamma, epsilon=epsilon)
            
            config_with_gamma = (size, n_goals, n_fires, p, gamma)
            times['value'][config_with_gamma] = []
            times['policy'][config_with_gamma] = []
            print(f'Config: {config_with_gamma}')
            
            for i in range(iter_):
                print(f'Iteration: {i+1}')
                start_time = time.time()
                if visualize:
                    visualize_iteration(maze, mdp, n=1, wait=.1, iteration_mode='value', timed_visualization=True, comparing=True)
                else:
                    mdp.value_iteration()
                times['value'][config_with_gamma].append((time.time() - start_time))
                mdp.reset()

                print(f'Value Iteration: {times['value'][config_with_gamma][-1]:.5f} (s)')

                start_time = time.time()
                if visualize:
                    visualize_iteration(maze, mdp, n=1, wait=.1, iteration_mode='policy', timed_visualization=True, comparing=True)
                else:
                    mdp.policy_iteration()
                times['policy'][config_with_gamma].append((time.time() - start_time))
                mdp.reset()
                
                print(f'Policy Iteration: {times['policy'][config_with_gamma][-1]:.5f} (s)')
                print('-'*30)

            avg_value_iteration_time = np.mean(times['value'][config_with_gamma])
            avg_policy_iteration_time = np.mean(times['policy'][config_with_gamma])
            print('==>')
            print(f'\tAvg Value Iteration: {avg_value_iteration_time:.5f} (s)')
            print(f'\tAvg Policy Iteration: {avg_policy_iteration_time:.5f} (s)')
            print('-'*30)

    if plot:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(maze_configs) * len(discount_factors))
        width = 0.15

        avg_times_value = []
        avg_times_policy= []
        avg_times_combined = []
        for size, n_goals, n_fires, p in maze_configs:
            for gamma in discount_factors:
                avg_times_value.append(np.mean(times['value'][(size, n_goals, n_fires, p, gamma)]))
                avg_times_policy.append(np.mean(times['policy'][(size, n_goals, n_fires, p, gamma)]))
                avg_times_combined.append((avg_times_value[-1]+avg_times_policy[-1])/2)

        ax.bar(x - width/2, avg_times_value, width, label='Value Iteration', color='mediumslateblue')
        ax.bar(x + width/2, avg_times_policy, width, label='Policy Iteration', color='teal')
        ax.plot(x, avg_times_combined, color='r', marker='o', linestyle='-', linewidth=2, alpha=.5, label='Average Runtime')

        ax.set_ylabel(f'Average Runtime Over {iter_} Iterations (s)')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{str(config)}, γ={gamma:.4f}' for config in maze_configs for gamma in discount_factors], rotation=45, ha='right')
        ax.legend()
        ax.grid(True)

        fig.tight_layout();
        plt.show();