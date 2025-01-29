from typing import List, Dict, Tuple
from enum import Enum
from maze import Maze, State, StateType

class ActionType(Enum):
    LEFT  = '<'
    RIGHT = '>'
    UP    = '^'
    DOWN  = 'v'
    STAY  = 's'

class Action:
    def __init__(self, maze: Maze, state: State, action_type: ActionType, noise: float = 0.2):
        self.maze = maze
        self.state = state
        self.action_type = action_type
        self.noise = noise
        self.intended_move, self.unintended_moves = self.getNextStates()
        self.next_states: List[Tuple[State, float]] = self.getValidNextStates()

    def __repr__(self):
        return f"Action(action={self.action_type})"

    def isValid(self, x: int, y: int) -> bool:
        return (0 <= x < self.maze.width and 0 <= y < self.maze.height) and \
            (self.maze.maze[y][x].state_type != StateType.WALL)

    def getNextStates(self) -> Tuple[Tuple[int, int], List[Tuple[int, int]]]:
        action_map = {
            ActionType.LEFT: ((self.state.x - 1, self.state.y),
                             [(self.state.x, self.state.y + 1), (self.state.x, self.state.y - 1)]),
            ActionType.RIGHT: ((self.state.x + 1, self.state.y),
                              [(self.state.x, self.state.y + 1), (self.state.x, self.state.y - 1)]),
            ActionType.UP: ((self.state.x, self.state.y - 1),
                           [(self.state.x + 1, self.state.y), (self.state.x - 1, self.state.y)]),
            ActionType.DOWN: ((self.state.x, self.state.y + 1),
                             [(self.state.x + 1, self.state.y), (self.state.x - 1, self.state.y)]),
            ActionType.STAY: ((self.state.x, self.state.y), [])
        }
        if self.action_type not in action_map:
            raise ValueError("Invalid action!")
        return action_map[self.action_type]

    def getValidNextStates(self) -> List[Tuple[State, float]]:
        next_states: List[Tuple[State, float]] = []
        x, y = self.intended_move

        if self.isValid(x, y):
            next_states.append((self.maze.maze[y][x], 1 - self.noise))

        for x, y in self.unintended_moves:
            if self.isValid(x, y):
                next_states.append((self.maze.maze[y][x], self.noise / 2))

        if not next_states:
            next_states.append((self.state, 1.0))

        return next_states

class MDP:
    def __init__(self,
                 maze: Maze,
                 states: List[State],
                 actions: Dict[State, List[Action]],
                 rewards: Dict[Tuple[State, Action, State], float],
                 discount_factor: float = .9,
                 epsilon: float = 1e-4):
        self.maze = maze
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        self.empty_states = []
        self.state_value_function: Dict[State, float] = {}
        self.policy = {}
        for state in self.states:
            if state.state_type == StateType.EMPTY:
                self.empty_states.append(state)
            self.state_value_function[state] = 0.
            self.policy[state] = Action(self.maze, state, ActionType.STAY)

    def __repr__(self):
        return f'MDP(discount_factor={self.discount_factor}, noise={self.noise:.3f})'
    
    def reset(self):
        self.empty_states = []
        for state in self.states:
            if state.state_type == StateType.EMPTY:
                self.empty_states.append(state)
            self.state_value_function[state] = 0.
            self.policy[state] = Action(self.maze, state, ActionType.STAY)

    def calculate_action_value(self, state: State, action: Action) -> float:
        return sum(
            prob * (self.rewards.get((state, action, next_state), 0) +
                    self.discount_factor * self.state_value_function[next_state])
            for next_state, prob in action.next_states
        )

    def value_iteration_step(self):
        delta = 0
        new_values = self.state_value_function.copy()

        for state in self.empty_states:
            action_values = [(self.calculate_action_value(state, action), action) 
                             for action in self.actions[state]]
            best_action_value, best_action = max(action_values, key=lambda x: x[0])
            new_values[state] = best_action_value
            self.policy[state] = best_action
            delta = max(delta, abs(new_values[state] - self.state_value_function[state]))

        self.state_value_function = new_values
        return delta

    def value_iteration(self):
        while True:
            delta = self.value_iteration_step()
            if delta < self.epsilon:
                break

    def extract_policy(self):
        for state in self.states:
            action_values = [(self.calculate_action_value(state, action), action) 
                             for action in self.actions[state]]
            _, best_action = max(action_values, key=lambda x: x[0])
            self.policy[state] = best_action

    def policy_evaluation(self, max_iterations=5):
        for _ in range(max_iterations):
            delta = 0
            for state in self.empty_states:
                action = self.policy[state]
                if action is None:
                    continue

                new_value = self.calculate_action_value(state, action)
                delta = max(delta, abs(new_value - self.state_value_function[state]))
                self.state_value_function[state] = new_value

            if delta < self.epsilon:
                break

    def policy_improvement(self):
        policy_stable = True

        for state in self.empty_states:
            old_action = self.policy[state]
            action_values = [(self.calculate_action_value(state, action), action) 
                             for action in self.actions[state]]

            best_action_value, best_action = max(action_values, key=lambda x: x[0])
            
            if best_action_value > self.state_value_function[state] and (old_action is None or old_action.action_type != best_action.action_type):
                self.policy[state] = best_action
                policy_stable = False

        return policy_stable
    
    def policy_iteration(self):
        while True:
            self.policy_evaluation()
            if self.policy_improvement():
                break

if __name__ == "__main__":
    ...
    # import time
    # start = time.time()
    # maze = Maze(width=28, height=16, p=0.0, n_goals=1, n_fires=1)
    # states = maze.maze.flatten()
    # end_maze = time.time()
    # print("States: ", states)

    # actions = {}
    # rewards = {}
    # for state in states:
    #     actions[state] = []
    #     for action_type in [ActionType.LEFT, ActionType.RIGHT, ActionType.UP, ActionType.DOWN]:
    #         action = Action(maze, state, action_type)
    #         actions[state].append(action)
    #         total_prob = sum(prob for _, prob in action.next_states)
    #         print(f'State({state.x}, {state.y}, Action({action.action_type})): Total probability - {total_prob}')
    #         for next_state, prob in action.next_states:
    #             if next_state.state_type == StateType.GOAL:
    #                 rewards[(action.state, action, next_state)] = 1.
    #             elif next_state.state_type == StateType.FIRE:
    #                 rewards[(action.state, action, next_state)] = -1.
    #             else:
    #                 rewards[(action.state, action, next_state)] = 0.0
    # end_init = time.time()
    # mdp = MDP(states, actions, rewards)
    # end_init2 = time.time()
    # mdp.value_iteration()
    # mdp.extract_policy()
    # end_iteration = time.time()
    # mdp.reset()
    # mdp.policy_iteration()
    # end_pol = time.time()
    # mdp.reset()
    
    # print("Maze Initialization time: ", end_maze - start)
    # print('Loop runtime: ', end_init - end_maze)
    # print('MDP initialization runtime: ', end_init2 - end_init)
    # print('Value iteration runtime: ', end_iteration - end_init2)
    # print('Policy iteration runtime: ', end_pol - end_iteration)