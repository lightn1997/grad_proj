
import numpy as np
from collections import defaultdict
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim


WHITE_DISK = 1
BLACK_DISK = -1
PROTAGONIST_TURN = 1
OPPONENT_TURN = -1


def copy_env(env, mute_env=True):
    new_env = env.__class__(
        board_size=env.board_size,
        sudden_death_on_invalid_move=env.sudden_death_on_invalid_move,
        mute=mute_env)
    new_env.reset()
    return new_env


class RandomPolicy(object):
    """Random policy for Othello."""

    def __init__(self, seed=0, q_values=None, epi=0):
        self.rnd = np.random.RandomState(seed=seed)
        self.env = None
        self.q_values = q_values
        self.num_episodes = epi

    def reset(self, env):
        if hasattr(env, 'env'):
            self.env = env.env
        else:
            self.env = env

    def seed(self, seed):
        self.rnd = np.random.RandomState(seed=seed)

    def get_action(self, obs):
        possible_moves = self.env.possible_moves
        ix = self.rnd.randint(0, max(0, len(possible_moves)))
        action = possible_moves[ix]
        return action
    
    def step(self, state, action, reward, next_state, done):
        pass

class GreedyPolicy(object):
    """Greed is good."""

    def __init__(self, q_values=None, epi=0):
        self.env = None
        self.q_values = q_values
        self.num_episodes = epi

    def reset(self, env):
        if hasattr(env, 'env'):
            self.env = env.env
        else:
            self.env = env

    def get_action(self, obs):
        my_perspective = self.env.player_turn
        new_env = copy_env(self.env)

        # For each move, replicate the current board and make the move.
        possible_moves = self.env.possible_moves
        disk_cnts = []
        for move in possible_moves:
            new_env.reset()
            new_env.set_board_state(
                board_state=obs, perspective=my_perspective)
            new_env.set_player_turn(my_perspective)
            new_env.step(move)
            white_disks, black_disks = new_env.count_disks()
            if my_perspective == WHITE_DISK:
                disk_cnts.append(white_disks)
            else:
                disk_cnts.append(black_disks)

        new_env.close()
        ix = np.argmax(disk_cnts)
        return possible_moves[ix]

    def step(self, state, action, reward, next_state, done):
        pass

class MaxiMinPolicy(object):
    """Maximin algorithm."""

    def __init__(self, max_search_depth=1, q_values=None, epi=0):
        self.env = None
        self.max_search_depth = max_search_depth
        self.q_values = q_values
        self.num_episodes = epi

    def reset(self, env):
        if hasattr(env, 'env'):
            self.env = env.env
        else:
            self.env = env

    def search(self, env, depth, perspective, my_perspective):

        # Search at a node stops if
        #   1. Game terminated
        #   2. depth has reached max_search_depth
        #   3. No more possible moves
        if (
                env.terminated or
                depth >= self.max_search_depth or
                len(env.possible_moves) == 0
        ):
            white_disks, black_disks = env.count_disks()
            if my_perspective == WHITE_DISK:
                return white_disks, None
            else:
                return black_disks, None
        else:
            assert env.player_turn == perspective
            new_env = copy_env(env)

            # For each move, replicate the current board and make the move.
            possible_moves = env.possible_moves
            disk_cnts = []
            for move in possible_moves:
                new_env.reset()
                new_env.set_board_state(env.get_observation(), env.player_turn)
                new_env.set_player_turn(perspective)
                new_env.step(move)
                if (
                        not new_env.terminated and
                        new_env.player_turn == perspective
                ):
                    # The other side had no possible moves.
                    new_env.set_player_turn(-perspective)
                disk_cnt, _ = self.search(
                    new_env, depth + 1, -perspective, my_perspective)
                disk_cnts.append(disk_cnt)

            new_env.close()

            # Max-min.
            ix = int(np.argmin(disk_cnts))
            if perspective == my_perspective:
                ix = int(np.argmax(disk_cnts))
            return disk_cnts[ix], possible_moves[ix]

    def get_action(self, obs):
        my_perspective = self.env.player_turn
        disk_cnt, move = self.search(env=self.env,
                                     depth=0,
                                     perspective=my_perspective,
                                     my_perspective=my_perspective)
        return move

    def step(self, state, action, reward, next_state, done):
        pass

class HumanPolicy(object):
    """Human policy."""

    def __init__(self, board_size, q_values=None, epi=0):
        self.board_size = board_size
        self.q_values = q_values
        self.num_episodes = epi

    def reset(self, env):
        pass

    def get_action(self, obs):
        return int(input('Enter action index:'))
    
    def step(self, state, action, reward, next_state, done):
        pass

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQN(object):
    def __init__(self, board_size, learned, q_values, epi):
        self.board_size = board_size
        self.learning_rate = 0.1
        self.QNetwork = QNetwork(board_size, board_size)
        self.optimizer = optim.Adam(self.QNetwork.parameters(), lr=self.learning_rate)
        self.discount_factor = 0.9
        self.exploration_prob = 1.0
        self.learned = learned
        self.q_values = q_values
        self.num_episodes = epi
        
    def reset(self, env):
        if hasattr(env, 'env'):
            self.env = env.env
        else:
            self.env = env

    def get_action(self, obs):
        possible_moves = self.env.possible_moves
        max_q_value = float('-inf')
        action = None
        if self.learned:
            with torch.no_grad():
                state_tensor = torch.tensor(obs, dtype=torch.float32)
                q_values = self.q_values(state_tensor)
                for act in possible_moves:
                    q = q_values[act // self.board_size][act % self.board_size]
                    if q > max_q_value:
                        max_q_value = q
                        action = act
        
        else:
            if random.uniform(0, 1) > self.exploration_prob:
                with torch.no_grad():
                    state_tensor = torch.tensor(obs, dtype=torch.float32)
                    q_values = self.QNetwork(state_tensor)
                    for act in possible_moves:
                        q = q_values[act // self.board_size][act % self.board_size]
                        if q > max_q_value:
                            max_q_value = q
                            action = act
            else :    
                action = random.choice(possible_moves)
        return action
    
    def step(self, state, action, reward, next_state, done):
        if self.learned :
            pass
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
            q_values = self.QNetwork(state_tensor)
            next_q_values = self.QNetwork(next_state_tensor)
            q_value = q_values[action // self.board_size][action % self.board_size]
            max_next_q_value = torch.max(next_q_values)
            target = q_value + self.discount_factor * max_next_q_value
            loss = nn.MSELoss()(q_value, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
            if done :
                self.q_values = self.QNetwork
                self.num_episodes += 1
                self.exploration_prob = 1 / math.log(self.num_episodes + 1 , 2)
