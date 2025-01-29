from typing import List, Tuple
from mdp import Action, ActionType, MDP
from maze import State, StateType, Maze
import matplotlib.pyplot as plt
import numpy as np
import pygame
import random

class EvaluateMaze:
    def __init__(self,
                 maze_configs: List[Tuple[Tuple[int, int], int, int, float]],
                 discount_factors: List[float],
                 noises: List[float],
                 n: int = 5,
                 decay_factor: float = 0.99,
                 epsilon: float = 0.00,
                 vwait: int = 625,
                 ptoggle: bool = False):
        self.maze_configs = maze_configs
        self.discount_factors = discount_factors
        self.noises = noises
        self.n = n
        self.decay_factor = decay_factor
        self.epsilon = epsilon
        self.vwait = vwait
        self.ptoggle = ptoggle

    def get_random_points(self, maze: Maze):
        points = [maze.get_random_empty_position() for _ in range(self.n)]
        return points

    def calculate_cumulative_reward(self, mdp: MDP, path: List[State]):
        """Calculate the cumulative reward along the path, considering the discount factor."""
        total_reward = 0.0
        discount = 1.0

        for i in range(len(path) - 1):
            state = path[i]
            next_state = path[i + 1]
            try:
                action = next(filter(lambda a, next_state=next_state: a.next_states[0][0] == next_state, mdp.actions[state]))  # Find the action leading to the next state
            except StopIteration:
                action = None
            if action is not None:
                reward = mdp.rewards.get((state, action, next_state), 0)
                total_reward += discount * reward
                discount *= mdp.discount_factor
            else:
                print(f'Warning: No action found from state {state} to {next_state}')

        return total_reward

    def follow_policy(self,
                      maze: Maze, 
                      mdp: MDP,
                      start_point: State,
                      noise: float,
                      visualize: bool = False):
        """Follow the policy from the start point and calculate total reward."""
        state = start_point
        path = [state]
        discount = mdp.discount_factor
        current_noise = noise
        epsilon = self.epsilon
        random_movement_count = 0

        max_steps = maze.width * maze.height // 5

        if visualize:
            pygame.init()
            screen = pygame.display.set_mode((maze.width * maze.cell_size,
                                              maze.height * maze.cell_size))

        step_count = 0

        while state and state.state_type == StateType.EMPTY and step_count < max_steps:
            if visualize:
                self.render_maze(maze, mdp, screen, state, path, start_point)

            if random.random() < epsilon:
                valid_actions = []
                for action_type in [ActionType.LEFT, ActionType.RIGHT,
                                    ActionType.UP, ActionType.DOWN]:
                    action = Action(maze, state, action_type, noise=current_noise)
                    if any(s.state_type != StateType.WALL for s, _ in action.next_states):
                        valid_actions.append(action)
                if valid_actions:
                    action = random.choice(valid_actions)
                    random_movement_count += 1
                else:
                    action = mdp.policy.get(state, Action(maze, state, ActionType.STAY, noise=current_noise))
            else:
                action = mdp.policy.get(state, Action(maze, state, ActionType.STAY, noise=current_noise))

            next_states = action.next_states
            if not next_states:
                break
        
            states, probabilities = zip(*next_states)
            if sum(probabilities) == 0:
                break

            next_state = random.choices(states, weights=probabilities, k=1)[0]
            if not next_state:
                print("Next state is None, breaking loop.")
                break

            state = next_state
            discount *= discount
            current_noise *= self.decay_factor
            epsilon *= self.decay_factor
            path.append(state)
            step_count += 1

        if self.ptoggle:
            print(f'Final decayed noise: {current_noise:.5f}, Random Actions: {random_movement_count:2d}')
        if visualize:
            pygame.quit()
        
        return path
    
    def calculate_expected_reward(self, mdp: MDP, start_point: State):
        """Calculate the expected reward following the policy without noise."""
        return mdp.state_value_function[start_point]
    
    def render_maze(self, 
                    maze: Maze, 
                    mdp: MDP, 
                    screen: pygame.Surface, 
                    agent_state: State, 
                    path: List[State], 
                    start_point: State):
        """Render the maze with current state values and directions."""
        def value_direction_visual(font, state: State):
            direction_text = font.render('', True, (0, 0, 0))
            if state.state_type == StateType.EMPTY:
                value_text = font.render(f'{mdp.state_value_function[state]:.2f}', True, (0, 0, 0))
                direction_text = font.render(str(mdp.policy[state].action_type.value), True, (0, 0, 0))
            elif state.state_type == StateType.GOAL:
                value_text = font.render('1.0', True, (0, 0, 0))
            elif state.state_type == StateType.FIRE:
                value_text = font.render('-1.0', True, (0, 0, 0))
            
            screen.blit(value_text, (state.x * maze.cell_size + 2, state.y * maze.cell_size + 2))
            screen.blit(direction_text, (state.x * maze.cell_size + maze.cell_size - 20,
                                         state.y * maze.cell_size - 23))
        
        font = pygame.font.SysFont(None, 30)
        screen.fill((255, 255, 255))
        for y in range(maze.height):
            for x in range(maze.width):
                cell = maze.maze[y][x]
                color = cell.color
                rect = pygame.Rect(x * maze.cell_size, y * maze.cell_size, maze.cell_size, maze.cell_size)
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, maze.colors['border'], rect, 1)
                if cell.state_type != StateType.WALL:
                    value_direction_visual(font, cell)

        for state in path:
            rect = pygame.Rect(state.x * maze.cell_size, state.y * maze.cell_size, maze.cell_size, maze.cell_size)
            pygame.draw.rect(screen, (123, 104, 238), rect)
            pygame.draw.rect(screen, maze.colors['border'], rect, 1)
            value_direction_visual(font, state)

        pygame.draw.rect(screen, (112, 128, 144), pygame.Rect(start_point.x * maze.cell_size, start_point.y * maze.cell_size, maze.cell_size, maze.cell_size))
        value_direction_visual(font, start_point)
        pygame.draw.rect(screen, maze.colors['border'], rect, 1)
        pygame.draw.rect(screen, (0, 0, 255), pygame.Rect(agent_state.x * maze.cell_size, agent_state.y * maze.cell_size, maze.cell_size, maze.cell_size))
        value_direction_visual(font, agent_state)
        pygame.draw.rect(screen, maze.colors['border'], rect, 1)
        pygame.display.flip()
        pygame.time.delay(self.vwait)

    def evaluate(self, 
                 iter_=100, 
                 visualize=False, 
                 plot=False, 
                 plot_title="Comparison of Rewards and Expected Utility", 
                 p_labels=['mconfigs', 'disc', 'noise']):
        """Evaluate the policy by comparing actual rewards to expected rewards."""
        accumulated_rewards_dict = {}
        print(f'** Experiment Started **\n- Total Iterations: {iter_}')
        print('-'*90)
        for config in self.maze_configs:
            size, n_goals, n_fires, p = config
            for gamma in self.discount_factors:
                for noise in self.noises:
                    maze = Maze(width=size[0], height=size[1], p=p, n_goals=n_goals, n_fires=n_fires)
                    states = maze.maze.flatten()

                    actions = {}
                    rewards = {}
                    for state in states:
                        actions[state] = []
                        for action_type in [ActionType.LEFT, ActionType.RIGHT,
                                            ActionType.UP, ActionType.DOWN]:
                            action = Action(maze, state, action_type, noise)
                            actions[state].append(action)
                            for next_state, _ in action.next_states:
                                if next_state.state_type == StateType.GOAL:
                                    rewards[(action.state, action, next_state)] = 1.
                                elif next_state.state_type == StateType.FIRE:
                                    rewards[(action.state, action, next_state)] =-1.
                                else:
                                    rewards[(action.state, action, next_state)] = 0.
                    
                    mdp = MDP(maze, states, actions, rewards, discount_factor=gamma)
                    x, y = size
                    if 15 < (x * y) * (1 - p) <= 1200:
                        mdp.policy_iteration()
                        print('<< Performed Policy Iteration >>')
                    else:
                        mdp.value_iteration()
                        print('<< Performed Value Iteration >>')

                    random_points = self.get_random_points(maze)
                    accumulated_rewards = []
                    print(f'Current Settings: Maze(size={size}, n_goals={n_goals}, n_fires={n_fires}, p={p}), γ={gamma:.2f},, noise={noise:.4f}')
                    print('-'*90)

                    pc_ = 1
                    for point in random_points:
                        total_rewards = []
                        for _ in range(iter_):
                            path = self.follow_policy(maze, mdp, point, noise=noise, visualize=visualize)
                            total_reward = self.calculate_cumulative_reward(mdp, path)
                            total_rewards.append(total_reward)

                        avg_reward = np.mean(total_rewards)
                        expected_reward = self.calculate_expected_reward(mdp, point)
                        accumulated_rewards.append((avg_reward, expected_reward))
                        print(f'Point: [{pc_}], Avg Reward: {avg_reward:.5}, Expected Reward: {expected_reward:.5f}, Diff: {np.abs(avg_reward - expected_reward):.5f}')
                        pc_ += 1
                    
                    accumulated_rewards_dict[(config, gamma, noise)] = accumulated_rewards
                    print('-'*90)

        if plot:
            fig, ax = plt.subplots(figsize=(20, 12))
            width = .15

            avg_rewards = []
            expected_rewards = []
            labels = []
            noise_colors = ['black', 'red', 'orange', 'brown', 'slategrey', 'royalblue', 'darkslateblue', 'dimgrey', 'cadatblue']
            color_idx = 0

            for config in self.maze_configs:
                color_idx = 0
                c_str = ''
                if 'mconfigs' in p_labels:
                    c_str += f'{config}, '
                for gamma in self.discount_factors:
                    g_str = ''
                    if 'disc' in p_labels:
                        g_str += f'{gamma}, '

                    for noise in self.noises:
                        n_str = ''
                        if 'noise' in p_labels:
                            n_str += str(noise)
                        accumulated_rewards = accumulated_rewards_dict[(config, gamma, noise)]
                        noise_avg_rewards = []
                        noise_expected_rewards = []
                        for ar in accumulated_rewards:
                            avg_rewards.append(ar[0])
                            expected_rewards.append(ar[1])
                            noise_avg_rewards.append(ar[0])
                            noise_expected_rewards.append(ar[1])
                            labels.append(c_str + g_str + n_str)
                        
                        x = np.arange(len(labels) - len(noise_avg_rewards), len(labels))
                        ax.plot(x, noise_avg_rewards, color=noise_colors[color_idx], linestyle='-', marker='o', label=f'Avg Reward (noise={noise})')
                        ax.plot(x, noise_expected_rewards, color=noise_colors[color_idx], linestyle='--', marker='x', label=f'Exp Reward (noise={noise})')
                        color_idx = (color_idx + 1) % len(noise_colors)

            x = np.arange(len(labels))
            ax.bar(x - width / 2, avg_rewards, width, label='Accumulated Reward', color='mediumslateblue')
            ax.bar(x + width / 2, expected_rewards, width, label='Expected Utility', color='teal')

            ax.set_ylabel('Reward')
            ax.set_title(plot_title)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=60, ha='right')
            
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            ax.grid(True)

            fig.tight_layout()
            plt.show()

if __name__ == '__main__':
    ...