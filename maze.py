from __future__ import annotations
from typing import Tuple, Dict, List
from enum import Enum
import numpy as np

__all__ = [
    'State',
    'Maze',
]

class StateType(Enum):
    EMPTY = 'empty'
    WALL  = 'wall'
    GOAL  = 'goal'
    FIRE  = 'fire'

class State:
    def __init__(self,
                 x: int, y: int,
                 state_type: str = 'empty',
                 color: Tuple[int, int, int] = (255, 255, 255)):
        self.x = x
        self.y = y
        self.state_type = StateType[state_type.upper()]
        self.color = color

    def __eq__(self, other):
        return (self.x == other.x) and (self.y == other.y)

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return f"State(x={self.x}, y={self.y}, state_type={self.state_type})"

class Maze:
    def __init__(self,
                 width: int,
                 height: int,
                 p: float = .66,
                 n_goals: int = 1,
                 n_fires: int = 1,
                 screen_width: int = 75,
                 screen_height: int = 75):
        self.width = width
        self.height = height
        self.p = p
        self.n_goals = n_goals
        self.n_fires = n_fires
        self.screen_width = screen_width * self.width
        self.screen_height = screen_height * self.height
        self.cell_size = min(self.screen_width // self.width,
                             self.screen_height // self.height)
        self.colors: Dict[str, Tuple[int, int, int]] = {
            'wall'  : (  0,   0,   0),
            'path'  : (255, 255, 255),
            'goal'  : (  0, 255,   0),
            'fire'  : (255,  30,   0),
            'border': (  0,   0,   0),
        }
        self.empty_states: List[State] = []
        self.goal_states: List[State] = []
        self.fire_states: List[State] = []
        self.maze = np.array([[State(x, y, state_type='empty') for x in range(self.width)] for y in range(self.height)])
        self.generate_random_maze(self.p)

    def generate_random_maze(self, p: float):
        for y in range(self.height):
            for x in range(self.width):
                if np.random.rand() < p:
                    self.maze[y][x] = State(x, y, state_type='wall', color=self.colors['wall'])
                else:
                    self.maze[y][x] = State(x, y, state_type='empty')
                    self.empty_states.append(self.maze[y][x])

        for _ in range(self.n_goals):
            goal = self.get_random_empty_position()
            if goal:
                goal.state_type = StateType.GOAL
                goal.color = self.colors['goal']
                self.goal_states.append(goal)

        for _ in range(self.n_fires):
            fire = self.get_random_empty_position()
            if fire:
                fire.state_type = StateType.FIRE
                fire.color = self.colors['fire']
                self.fire_states.append(fire)
                    
    def get_random_empty_position(self) -> State:
        if not self.empty_states:
            return None
        new_pos = np.random.choice(self.empty_states)
        self.empty_states.remove(new_pos)
        return new_pos

    def restore_original_maze(self):
        for y in range(self.height):
            for x in range(self.width):
                if self.maze[y][x].state_type == StateType.EMPTY:
                    self.maze[y][x].color = self.colors['path']
                elif self.maze[y][x].state_type == StateType.WALL:
                    self.maze[y][x].color = self.colors['wall']

        self.empty_states = [self.maze[y][x] for y in range(self.height) for x in range(self.width) if self.maze[y][x].state_type == StateType.EMPTY]

        start_position = self.get_random_empty_position()
        if start_position:
            return start_position, False
        else:
            raise ValueError('Failed to find a valid starting position')
        