from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from mdp import Action, ActionType
from maze import State, StateType, Maze
from typing import List, Tuple
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime
from enum import Enum
from utils import moving_average
import numpy as np
import random
import pygame
import time
import os

__all__ = [
    'LinearDQNetwork',
    'CNNDQNetwork',
    'ModelType',
    'ReplayBuffer',
    'DQLearning',
]

def ensure_directory(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)

class LinearDQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, n_neuron: int = 128):
        super(LinearDQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, n_neuron)
        self.fc2 = nn.Linear(n_neuron, n_neuron)
        self.fc3 = nn.Linear(n_neuron, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class CNNDQNetwork(nn.Module):
    def __init__(self, input_shape, output_dim, n_neuron):
        super(CNNDQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        conv_output_size = self._get_conv_output(input_shape)
        self.fc1 = nn.Linear(conv_output_size, n_neuron)
        self.fc2 = nn.Linear(n_neuron, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    
    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.zeros(1, *shape)
            output = self._forward_conv(input)
            return int(np.prod(output.size()))
        
    def _forward_conv(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return x

class ModelType(Enum):
    CNN = CNNDQNetwork
    LINEAR = LinearDQNetwork

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

class DQLearning:
    def __init__(self,
                 maze: Maze,
                 actions: List[ActionType],
                 model: ModelType,
                 /,
                 *,
                 n_neuron: int = 256,
                 discount_factor: float = 0.9,
                 learning_rate: float = 0.001,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 replay_buffer_capacity: int = 100000,
                 batch_size: int = 64,
                 render_: bool = False,
                 vwait: int = 15):
        self.maze = maze
        self.actions = actions
        self.model = model
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.render_ = render_
        self.vwait = vwait
        self.max_s = self.maze.width * self.maze.height * (1 - self.maze.p + 0.2)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        action_dim = len(self.actions)
        if self.model == ModelType.CNN:
            input_shape = (1, self.maze.height, self.maze.width)
            self.policy_net = CNNDQNetwork(input_shape, action_dim, n_neuron).to(self.device)
            self.target_net = CNNDQNetwork(input_shape, action_dim, n_neuron).to(self.device)
            print('PolicyNet / TargetNet Model:')
            summary(self.policy_net, input_shape, self.batch_size, str(self.device))
        else:
            state_dim = self.maze.width * self.maze.height
            self.policy_net = LinearDQNetwork(state_dim, action_dim, n_neuron).to(self.device)
            self.target_net = LinearDQNetwork(state_dim, action_dim, n_neuron).to(self.device)
            print('PolicyNet / TargetNet Model:')
            summary(self.policy_net, (state_dim,), self.batch_size, str(self.device))

        self.optim = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.action_to_index = {}
        self.index_to_action = {}
        for i, action in enumerate(actions):
            self.action_to_index[action] = i
            self.index_to_action[i] = action

        self.path = []

        if self.render_:
            pygame.init()
            self.screen = pygame.display.set_mode((
                self.maze.width * self.maze.cell_size, self.maze.height * self.maze.cell_size
            ))

    def get_state_representation(self, state: State) -> np.ndarray:
        if self.model == ModelType.CNN:
            state_rep = np.zeros((1, self.maze.height, self.maze.width))
            state_rep[0, state.y, state.x] = 1
            for goal in self.maze.goal_states:
                state_rep[0, goal.y, goal.x] = 2
            for fire in self.maze.fire_states:
                state_rep[0, fire.y, fire.x] = 3
            return state_rep
        elif self.model == ModelType.LINEAR:
            state_vector = np.zeros(self.maze.width * self.maze.height)
            state_vector[state.y * self.maze.width + state.x] = 1
            return state_vector

    def choose_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.choice(range(len(self.actions)))
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return np.argmax(q_values).item()
    
    def optimizer(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = self.replay_buffer.sample(self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        action_batch = torch.LongTensor(action_batch).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + (self.discount_factor * next_q_values * (1 - done_batch))

        loss = self.criterion(q_values, expected_q_values.unsqueeze(1))
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def validate(self, num_episodes: int = 20) -> Tuple[float, float]:
        total_rewards = 0
        success_count = 0
        original_epsilon = self.epsilon
        self.epsilon = 0
        for _ in range(num_episodes):
            state, _ = self.maze.restore_original_maze()
            steps = 0

            while state.state_type == StateType.EMPTY:
                action_index = self.choose_action(self.get_state_representation(state))
                action = self.index_to_action[action_index]
                next_state, reward = self.take_action(state, action)
                state = next_state
                steps += 1

                if state.state_type == StateType.GOAL:
                    success_count += 1
                    break
                if state.state_type == StateType.FIRE or steps >= self.max_s:
                    break
            total_rewards += reward
        
        self.epsilon = original_epsilon
        accuracy = success_count / num_episodes
        avg_reward = total_rewards / num_episodes
        return accuracy, avg_reward

    def train(self, 
              episodes: int = 1e3, 
              plot: bool = True, 
              improvement_threshold: float = 1e-5,
              patience: int = 10):
        print(f'Training Started ({self.model})')
        print('-'*90)
        episode_rewards = []
        validation_accuracy = []
        validation_rewards = []
        best_mean_reward = -np.inf
        num_no_improvement_episodes = 0
        start_tr_time = time.time()

        for episode in range(episodes):
            state, _ = self.maze.restore_original_maze()
            self.path = []
            steps = 0
            episode_reward = 0

            while state.state_type == StateType.EMPTY:
                action_index = self.choose_action(self.get_state_representation(state))
                action = self.index_to_action[action_index]
                next_state, reward = self.take_action(state, action)
                done = next_state.state_type in [StateType.GOAL, StateType.FIRE]
                self.replay_buffer.push(self.get_state_representation(state), action_index, reward, self.get_state_representation(next_state), done)
                self.path.append(state)
                state = next_state
                episode_reward += reward
                steps += 1

                self.optimizer()

                if done or steps == self.max_s:
                    break
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            episode_rewards.append(episode_reward)

            if (episode + 1) % 25 == 0:
                end_tr_time = time.time()
                self.target_net.load_state_dict(self.policy_net.state_dict())
                
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
            smoothed_validation_rewrads = moving_average(validation_rewards, window_size=1)

            plt.subplot(3, 1, 1)
            plt.plot(np.arange(len(smoothed_rewards)) + 100, smoothed_rewards, c='mediumslateblue')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Training Progress')
            plt.grid(True)

            plt.subplot(3, 1, 2)
            plt.plot(np.arange(len(smoothed_validation_accuracy)) * 100 + 100, smoothed_validation_accuracy, c='salmon')
            plt.xlabel('Episode')
            plt.ylabel('Validation Accuracy')
            plt.title('Validation Accuracy During Training')
            plt.grid(True)

            plt.subplot(3, 1, 3)
            plt.plot(np.arange(len(smoothed_validation_rewrads)) * 100 + 100, smoothed_validation_rewrads, c='lightseagreen')
            plt.xlabel('Episode')
            plt.ylabel('Validation Avg Reward')
            plt.title('Validation Avg Reward During Training')
            plt.grid(True)

            plt.tight_layout()
            plt.show()
            
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
            return state, -0.25
        
        next_state = self.maze.maze[y][x]
        reward_table = {
            StateType.GOAL: (next_state, 1.0),
            StateType.FIRE: (next_state, -1.0),
            StateType.WALL: (state, -0.25),
            StateType.EMPTY: (next_state, -0.005)
        }

        reward = reward_table[next_state.state_type]
        return reward[0], reward[1]
    
    def evaluate(self, num_episodes: int = 100, render_: bool = True):
        print(f'Evaluation Started ({self.model})')
        total_rewards = []
        success_count = 0
        steps_list = []
        original_epsilon = self.epsilon
        self.render_ = render_
        self.epsilon = 0
        
        for _ in range(num_episodes):
            state, _ = self.maze.restore_original_maze()
            self.path = [state]
            if self.render_:
                self.render(state)
            episode_reward = 0
            steps = 0
            
            while state.state_type == StateType.EMPTY:
                action_index = self.choose_action(self.get_state_representation(state))
                action = self.index_to_action[action_index]
                next_state, reward = self.take_action(state, action)
                state = next_state
                self.path.append(state)
                episode_reward += reward
                steps += 1

                if self.render_:
                    self.render(state)
                    pygame.time.wait(self.vwait+20)

                if state.state_type in [StateType.GOAL, StateType.FIRE] or steps >= self.max_s:
                    if state.state_type == StateType.GOAL:
                        success_count += 1
                    break
            
            total_rewards.append(episode_reward)
            steps_list.append(steps)
        
        self.epsilon = original_epsilon
        # print(f'Evaluation completed in {steps} steps with total reward {total_reward}')

        avg_reward = np.mean(total_rewards)
        success_rate = success_count / num_episodes
        avg_steps_to_goal = np.mean(steps_list)

        print(f'Evaluation over {num_episodes} episodes:')
        print(f'Average Reward: {avg_reward}')
        print(f'Success Rate: {success_rate * 100}%')
        print(f'Average Steps to Goal: {avg_steps_to_goal}')

        return success_rate, avg_reward, avg_steps_to_goal

    def render(self, agent_state: State):

        def value_direction_visual(font, state: State, best_action: ActionType):
            value_text = font.render('0.0', True, (0, 0, 0))
            direction_text = font.render('', True, (0, 0, 0))
            if state.state_type == StateType.EMPTY:
                state_rep = self.get_state_representation(state)
                state_tensor = torch.FloatTensor(state_rep).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = self.policy_net(state_tensor)
                q_value = q_values.max().item()
                best_action_index = torch.argmax(q_values).item()
                best_action = self.index_to_action[best_action_index]
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
        for y in range(self.maze.height):
            for x in range(self.maze.width):
                cell = self.maze.maze[y][x]
                color = cell.color
                rect = pygame.Rect(x * self.maze.cell_size, y * self.maze.cell_size, self.maze.cell_size, self.maze.cell_size)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.maze.colors['border'], rect, 1)
                if cell.state_type != StateType.WALL:
                    state_rep = self.get_state_representation(cell)
                    state_tensor = torch.FloatTensor(state_rep).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        q_values = self.policy_net(state_tensor)
                    best_action_index = torch.argmax(q_values).item()
                    best_action = self.index_to_action[best_action_index]
                    value_direction_visual(font, cell, best_action)

        for state in self.path:
            rect = pygame.Rect(state.x * self.maze.cell_size, state.y * self.maze.cell_size,
                               self.maze.cell_size, self.maze.cell_size)
            pygame.draw.rect(self.screen, (123, 104, 238), rect)
            pygame.draw.rect(self.screen, self.maze.colors['border'], rect, 1)
            state_rep = self.get_state_representation(state)
            state_tensor = torch.FloatTensor(state_rep).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            best_action_index = torch.argmax(q_values).item()
            best_action = self.index_to_action[best_action_index]
            value_direction_visual(font, state, best_action)
            pygame.display.flip()
            pygame.time.wait(self.vwait-5)

        rect = pygame.Rect(agent_state.x * self.maze.cell_size, agent_state.y * self.maze.cell_size,
                           self.maze.cell_size, self.maze.cell_size)
        pygame.draw.rect(self.screen, (0, 0, 255), rect)
        pygame.draw.rect(self.screen, self.maze.colors['border'], rect, 1)
        state_rep = self.get_state_representation(agent_state)
        state_tensor = torch.FloatTensor(state_rep).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        best_action_index = torch.argmax(q_values).item()
        best_action = self.index_to_action[best_action_index]
        value_direction_visual(font, agent_state, best_action)
        pygame.display.flip()
        pygame.time.wait(self.vwait)

    def load_model(self, model_path: str):
        self.policy_net.load_state_dict(torch.load(model_path, weights_only=True))
        self.policy_net.eval()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        print(f'Model loaded from {model_path}')

    def save_model(self, model_path: str):
        torch.save(self.policy_net.state_dict(), model_path)
        print(f'Model saved to {model_path}')

    def save_maze(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        maze_dir = 'MazeLayouts'
        ensure_directory(maze_dir)
        maze_path = os.path.join(maze_dir, f'maze_{self.maze.width}_{self.maze.height}_{timestamp}.txt')
        model_path = os.path.join(maze_dir, f'model_{self.maze.width}_{self.maze.height}_{timestamp}.pth')

        with open(maze_path, 'w') as file:
            file.write(f'{self.maze.width} {self.maze.height}\n')
            for y in range(self.maze.height):
                for x in range(self.maze.width):
                    state = self.maze.maze[y][x]
                    file.write(f'{state.x} {state.y} {state.state_type.value}\n')
        print(f'Maze saved to {maze_path}')

        torch.save(self.policy_net.state_dict(), model_path)
        print(f'Model saved to {model_path}')

    def load_maze(self, maze_path: str, model_path: str):
        with open(maze_path, 'r') as file:
            lines = file.readlines()
            width, height = map(int, lines[0].strip().split())
            self.maze = Maze(width, height)
            self.maze.goal_states = []
            self.maze.fire_states = []
            for line in lines[1:]:
                x, y, state_type = line.strip().split()
                x, y = int(x), int(y)
                state = State(x, y, state_type=state_type.lower())
                self.maze.maze[y][x] = state
                if state.state_type == StateType.GOAL:
                    state.color = self.maze.colors['goal']
                    self.maze.goal_states.append(state)
                elif state.state_type == StateType.FIRE:
                    state.color = self.maze.colors['fire']
                    self.maze.fire_states.append(state)
        
        self.load_model(model_path)
        print(f'Maze loaded from {maze_path} with model {model_path}')

if __name__ == '__main__':
    maze = Maze(width=15, height=15, p=0.2, n_goals=2, n_fires=2, screen_width=75, screen_height=55)
    # dql_cnn = DQLearning(maze,
    #                  [ActionType.LEFT, ActionType.RIGHT,
    #                   ActionType.UP, ActionType.DOWN],
    #                  ModelType.CNN,
    #                  learning_rate = 0.01,
    #                  discount_factor=0.9,
    #                  batch_size=64,
    #                  render_=True,
    #                  vwait=0)
    
    dql_linear = DQLearning(maze,
                     [ActionType.LEFT, ActionType.RIGHT,
                      ActionType.UP, ActionType.DOWN],
                     ModelType.LINEAR,
                     learning_rate = 0.01,
                     discount_factor=0.9,
                     batch_size=64,
                     render_=True,
                     vwait=0)

    # dql_cnn.train(episodes=5000, improvement_threshold=1e-4, patience=10)
    # dql_cnn.save_maze()

    dql_linear.train(episodes=1500, improvement_threshold=1e-4, patience=20)
    dql_linear.save_maze()

    # dql_cnn.evaluate(1000, False)

    dql_linear.evaluate(250, False)


    # dql.load_maze('MazeLayouts/maze_20250118_043603.txt',
    #               'MazeLayouts/model_20250118_043603.pth')
    # dql.evaluate(100)    
    # dql.train(episodes=300)