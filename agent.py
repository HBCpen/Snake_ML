import torch
import random
import numpy as np
from collections import deque

from game import SnakeGame, Direction, Point
from model import CombinedNet, QTrainer
from constants import *

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0.0  # 探索率 (ランダムな行動をとる確率)
        self.gamma = 0.9  # 割引率
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = None # モデルはtrain.pyで初期化
        self.trainer = None # トレーナーもtrain.pyで初期化

    def get_state(self, game, snake_id):
        snake = game.snake1 if snake_id == 1 else game.snake2
        other_snake = game.snake2 if snake_id == 1 else game.snake1

        # 1. グリッドデータ (チャンネル)
        food_channel = np.zeros((game.w, game.h), dtype=np.float32)
        for pt in game.food:
            food_channel[pt.x, pt.y] = 1

        wall_channel = np.zeros((game.w, game.h), dtype=np.float32)
        for pt in game.walls:
            wall_channel[pt.x, pt.y] = 1

        my_head_channel = np.zeros((game.w, game.h), dtype=np.float32)
        if 0 <= snake.head.x < game.w and 0 <= snake.head.y < game.h:
            my_head_channel[snake.head.x, snake.head.y] = 1

        my_body_channel = np.zeros((game.w, game.h), dtype=np.float32)
        for pt in snake.body[1:]:
            if 0 <= pt.x < game.w and 0 <= pt.y < game.h:
                my_body_channel[pt.x, pt.y] = 1

        opp_head_channel = np.zeros((game.w, game.h), dtype=np.float32)
        if 0 <= other_snake.head.x < game.w and 0 <= other_snake.head.y < game.h:
            opp_head_channel[other_snake.head.x, other_snake.head.y] = 1

        opp_body_channel = np.zeros((game.w, game.h), dtype=np.float32)
        for pt in other_snake.body[1:]:
            if 0 <= pt.x < game.w and 0 <= pt.y < game.h:
                opp_body_channel[pt.x, pt.y] = 1

        # チャンネルをスタック
        grid_state = np.stack([
            food_channel,
            wall_channel,
            my_head_channel,
            my_body_channel,
            opp_head_channel,
            opp_body_channel
        ], axis=0)

        # 2. ベクトルデータ
        dir_l = snake.direction == Direction.LEFT
        dir_r = snake.direction == Direction.RIGHT
        dir_u = snake.direction == Direction.UP
        dir_d = snake.direction == Direction.DOWN
        my_dir_vec = [dir_l, dir_r, dir_u, dir_d]

        other_dir_l = other_snake.direction == Direction.LEFT
        other_dir_r = other_snake.direction == Direction.RIGHT
        other_dir_u = other_snake.direction == Direction.UP
        other_dir_d = other_snake.direction == Direction.DOWN
        opp_dir_vec = [other_dir_l, other_dir_r, other_dir_u, other_dir_d]

        # 最寄りの餌へのベクトル
        nearest_food_vec = [0, 0]
        if game.food:
            distances = [np.linalg.norm(np.array(snake.head) - np.array(f)) for f in game.food]
            nearest_food_idx = np.argmin(distances)
            nearest_food = game.food[nearest_food_idx]
            nearest_food_vec = (np.array(nearest_food) - np.array(snake.head)) / np.array([game.w, game.h])

        vector_state = np.concatenate([
            # 長さ (正規化)
            [snake.length / (game.w * game.h)],
            [other_snake.length / (game.w * game.h)],
            # 進行方向 (one-hot)
            my_dir_vec,
            opp_dir_vec,
            # 餌へのベクトル
            nearest_food_vec
        ]).astype(np.float32)

        grid_tensor = torch.from_numpy(grid_state).unsqueeze(0).to(DEVICE)
        vector_tensor = torch.from_numpy(vector_state).unsqueeze(0).to(DEVICE)

        return grid_tensor, vector_tensor

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # ε-greedy法による探索と活用のトレードオフ
        self.epsilon = 80 - self.n_games # ゲーム回数を重ねるごとに探索率を下げる
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state_grid, state_vector = state
            prediction = self.model(state_grid, state_vector)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
