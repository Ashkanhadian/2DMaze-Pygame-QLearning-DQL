from maze import Maze, State, StateType
from mdp import Action, ActionType
from typing import List, Tuple, Dict
from utils import moving_average
import matplotlib.pyplot as plt
import numpy as np
import pygame
import time

class QLearning:
    def __init__(self,
                 maze: Maze,
                 actions: List[ActionType],
                 discount_factor: float = .9,
                 alpha: float = .4,
                 epsilon: float = 1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 render_: bool = False,
                 vwait: int = 1):
        self.maze = maze
        self.actions = actions
        self.discount_factor = discount_factor
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {(state.x, state.y): {action: 0 for action in actions} 
                        for state in maze.maze.flatten()}
        self.path: List[State] = []
        self.render_ = render_
        self.vwait = vwait
        self.max_s = self.maze.width * self.maze.height * (1 - self.maze.p)
    
        if self.render_:
            pygame.init()
            self.screen = pygame.display.set_mode((
                self.maze.width * self.maze.cell_size, self.maze.height * self.maze.cell_size
            ))

    def choose_action(self, state: State) -> ActionType:
        if state.state_type != StateType.EMPTY:
            return ActionType.STAY
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        return max(self.q_table[(state.x, state.y)], key=self.q_table[(state.x, state.y)].get)

    def update_q_value(self, 
                       state: State, 
                       action: ActionType, 
                       reward: float, 
                       next_state: State):
        if state.state_type != StateType.EMPTY:
            return
        old_q_value = self.q_table[(state.x, state.y)][action]
        next_max = max(self.q_table[(next_state.x, next_state.y)].values())
        new_q_value = old_q_value + self.alpha * (reward + self.discount_factor * next_max - old_q_value)
        self.q_table[(state.x, state.y)][action] = new_q_value
        # print(f'Updated Q-value for state ({state.x}, {state.y}) with action {action}: {new_q_value}')

    def validate(self, num_episodes: int = 20) -> Tuple[float, float]:
        total_rewards = 0
        success_count = 0
        original_epsilon = self.epsilon
        self.epsilon = 0

        for _ in range(num_episodes):
            state, _ = self.maze.restore_original_maze()
            steps = 0
            episode_reward = 0

            while state.state_type == StateType.EMPTY:
                action = self.choose_action(state)
                next_state, reward = self.take_action(state, action)
                state = next_state
                episode_reward += reward
                steps += 1
    
                if state.state_type == StateType.GOAL:
                    success_count += 1
                    break

                if state.state_type == StateType.FIRE or steps >= self.max_s:
                    break

            total_rewards += episode_reward

        self.epsilon = original_epsilon
        accuracy = success_count / num_episodes
        avg_reward = total_rewards / num_episodes
        return accuracy, avg_reward

    def train(self, 
              episodes: int = 1e3,
              plot: bool = True, 
              improvement_threshold: float =1e-4, 
              patience: int = 10):
        print('Training Started')
        print('-'*90)
        episode_rewards = []
        best_mean_reward = -np.inf
        num_no_improvement_episodes = 0
        validation_accuracy = []
        validation_rewards = []
        start_tr_time = time.time()

        for episode in range(episodes):
            state, _ = self.maze.restore_original_maze()
            self.path = []
            steps = 0
            episode_reward = 0

            while state.state_type == StateType.EMPTY:
                action = self.choose_action(state)
                next_state, reward = self.take_action(state, action)
                self.update_q_value(state, action, reward, next_state)
                self.path.append(state)
                episode_reward += reward
                state = next_state
                steps += 1

                if next_state.state_type in [StateType.GOAL, StateType.FIRE] or steps >= self.max_s:
                    break
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            episode_rewards.append(episode_reward)

            if (episode + 1) % 25 == 0:
                end_tr_time = time.time()
                recent_mean_reward = np.mean(episode_rewards[-25:])
                if recent_mean_reward > best_mean_reward + improvement_threshold:
                    best_mean_reward = recent_mean_reward
                    num_no_improvement_episodes = 0
                else:
                    num_no_improvement_episodes += 1

                if episode < 100:
                    num_episodes = 20
                elif 100 <= episode < 200:
                    num_episodes = 50
                else:
                    num_episodes = 100

                accuracy, avg_reward = self.validate(num_episodes=num_episodes)
                validation_accuracy.append(accuracy)
                validation_rewards.append(avg_reward)
                print(f'Episode [{episode + 1:04}], Completed in {steps:3} steps, Training Reward {episode_reward}\nValidation Accuracy: {accuracy * 100:.2f}%, Validation Avg Reward: {avg_reward:.2f}, Last 25 episodes runtime: {end_tr_time-start_tr_time:2.4f} (s)')
                print('-'*90)
                start_tr_time = time.time()

                if num_no_improvement_episodes >= patience:
                    print('Early stopping triggered. Training stopped.')
                    break

            if self.render_:
                self.render(state)
                pygame.time.wait(self.vwait)
        if plot:
            smoothed_rewards = moving_average(episode_rewards, window_size=100)
            smoothed_validation_accuracy = moving_average(validation_accuracy, window_size=1)
            smoothed_validation_rewards = moving_average(validation_rewards, window_size=1)

            plt.subplot(3, 1, 1)
            plt.plot(np.arange(len(smoothed_rewards)) + 100, smoothed_rewards, c='mediumslateblue')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Total Reward During Training')

            plt.subplot(3, 1, 2)
            plt.plot(np.arange(len(smoothed_validation_accuracy)) * 100 + 100, smoothed_validation_accuracy, c='salmon')
            plt.xlabel('Episode')
            plt.ylabel('Validation Accuracy')
            plt.title('Validation Accuracy During Training')

            plt.subplot(3, 1, 3)
            plt.plot(np.arange(len(smoothed_validation_rewards)) * 100 + 100, smoothed_validation_rewards, c='lightseagreen')
            plt.xlabel('Episode')
            plt.ylabel('Validation Avg Reward')
            plt.title('Validation Avg Reward During Training')

            plt.tight_layout()
            plt.show()

    def evaluate(self, num_episodes: int = 100):
        print('Evaluation Started')
        print('-'*90)
        total_rewards = []
        success_count = 0
        steps_list = []

        original_epsilon = self.epsilon
        self.epsilon = 0
        for episode in range(num_episodes):
            state, _ = self.maze.restore_original_maze()
            self.path = [state]
            if self.render_:
                self.render(state)
                pygame.time.wait(self.vwait)
            episode_reward = 0
            steps = 0

            while state.state_type == StateType.EMPTY:
                action = self.choose_action(state)
                next_state, reward = self.take_action(state, action)
                state = next_state
                self.path.append(state)
                if self.render_:
                    self.render(state)
                    pygame.time.wait(self.vwait)
                episode_reward += reward
                steps += 1

                if state.state_type in [StateType.GOAL, StateType.FIRE] or steps >= self.max_s:
                    if state.state_type == StateType.GOAL:
                        success_count += 1
                    break
            
            total_rewards.append(episode_reward)
            steps_list.append(steps)

            if (episode + 1) % 10 == 0:
                print(f'Episode {episode + 1}/{num_episodes} completed')

        self.epsilon = original_epsilon
        avg_reward = np.mean(total_rewards)
        success_rate = success_count / num_episodes
        avg_steps_to_goal = np.mean(steps_list)

        print(f'Evaluation over {num_episodes} episodes:')
        print(f'Average Reward: {avg_reward}')
        print(f'Success Rate: {success_rate * 100}%')
        print(f'Average Steps to Goal: {avg_steps_to_goal}')

        return success_rate, avg_reward, avg_steps_to_goal

    def take_action(self, state: State, action_type: ActionType) -> Tuple[State, float]:
        if state.state_type == StateType.WALL:
            return state, 0
        
        action = Action(self.maze, state, action_type)
        intended_next_state = action.intended_move
        if np.random.rand() > action.noise or not action.unintended_moves:
            next_state_coords = intended_next_state
        else:
            next_state_coords = action.unintended_moves[np.random.randint(len(action.unintended_moves))]
        
        x, y = next_state_coords
        if x < 0 or x >= self.maze.width or y < 0 or y >= self.maze.height:
            return state, 0

        next_state = self.maze.maze[y][x]
        reward_table = {
            StateType.GOAL: (next_state, 1.0),
            StateType.FIRE: (next_state, -1.0),
            StateType.WALL: (state, -0.5),
            StateType.EMPTY: (next_state, -0.01)
        }

        reward = reward_table[next_state.state_type]
        return reward[0], reward[1]

    def extract_policy(self) -> Dict[Tuple[int, int], ActionType]:
        policy = {}
        for (x, y), actions in self.q_table.items():
            best_action = max(actions, key=actions.get)
            policy[(x, y)] = best_action
        return policy

    def render(self, agent_state: State):

        def value_direction_visual(font, state: State, best_action: ActionType):
            value_text = font.render('0.0', True, (0, 0, 0))
            direction_text = font.render('', True, (0, 0, 0))
            if state.state_type == StateType.EMPTY:
                q_value = max(self.q_table.get((state.x, state.y), {}).values(), default=0)
                value_text = font.render(f'{q_value:.2f}', True, (0, 0, 0))
                direction_text = font.render(best_action.value, True, (0, 0, 0))
            elif state.state_type == StateType.GOAL:
                value_text = font.render('1.0', True, (0, 0, 0))
            elif state.state_type == StateType.FIRE:
                value_text = font.render('-1.0', True, (0, 0, 0))
            else:
                value_text = font.render('0.0', True, (0, 0, 0))
                direction_text = font.render('', True, (0, 0, 0))

            self.screen.blit(value_text, (state.x * self.maze.cell_size + 2, 
                                          state.y * self.maze.cell_size + 2))
            self.screen.blit(direction_text, (state.x * self.maze.cell_size + self.maze.cell_size - 20,
                                              state.y * self.maze.cell_size + 23))
        
        font = pygame.font.SysFont(None, 30)
        self.screen.fill((255, 255, 255))
        best_actions = self.extract_policy()

        for y in range(self.maze.height):
            for x in range(self.maze.width):
                cell = self.maze.maze[y][x]
                color = cell.color
                rect = pygame.Rect(x * self.maze.cell_size, y * self.maze.cell_size,
                                   self.maze.cell_size, self.maze.cell_size)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.maze.colors['border'], rect, 1)
                if cell.state_type != StateType.WALL:
                    best_action = best_actions.get((x, y), ActionType.STAY)
                    value_direction_visual(font, cell, best_action)
        
        for state in self.path:
            rect = pygame.Rect(state.x * self.maze.cell_size, state.y * self.maze.cell_size, 
                               self.maze.cell_size, self.maze.cell_size)
            pygame.draw.rect(self.screen, (123, 104, 238), rect)
            pygame.draw.rect(self.screen, self.maze.colors['border'], rect, 1)
            best_action = best_actions.get((state.x, state.y), ActionType.STAY)
            value_direction_visual(font, state, best_action)

        rect = pygame.Rect(agent_state.x * self.maze.cell_size, agent_state.y * self.maze.cell_size,
                            self.maze.cell_size, self.maze.cell_size)
        pygame.draw.rect(self.screen, (0, 0, 255), rect)
        pygame.draw.rect(self.screen, self.maze.colors['border'], rect, 1)
        best_action = best_actions.get((agent_state.x, agent_state.y), ActionType.STAY)
        value_direction_visual(font, agent_state, best_action)
        pygame.display.flip()
        # pygame.time.wait(self.vwait)

if __name__ == '__main__':
    maze = Maze(width=13, height=10, p=0.2, n_goals=2, n_fires=1, screen_width=75, screen_height=55)
    q_learning = QLearning(maze, 
                           actions=[ActionType.LEFT, ActionType.RIGHT,
                                    ActionType.UP, ActionType.DOWN], 
                           alpha=0.1,
                           render_=True,
                           vwait=0)
    q_learning.train(episodes=10000, patience=50)

    q_learning.evaluate(num_episodes=100)