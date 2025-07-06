import pygame
import random
from enum import Enum
from collections import namedtuple

from constants import *

# Pygameの初期化
pygame.init()
font = pygame.font.Font(None, 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

class Snake:
    def __init__(self, initial_pos, color, head_color):
        self.color = color
        self.head_color = head_color
        self.initial_pos = initial_pos
        self.reset()

    def reset(self):
        self.head = Point(self.initial_pos[0], self.initial_pos[1])
        self.body = [self.head,
                     Point(self.head.x - 1, self.head.y),
                     Point(self.head.x - 2, self.head.y)]
        self.direction = Direction.RIGHT
        self.length = len(self.body)

class SnakeGame:
    def __init__(self, w=GRID_WIDTH, h=GRID_HEIGHT):
        self.w = w
        self.h = h
        # Pygameディスプレイのセットアップ
        self.display = pygame.display.set_mode((self.w * GRID_SIZE, self.h * GRID_SIZE))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # スネークの初期化
        self.snake1 = Snake(initial_pos=(5, 5), color=SNAKE1_COLOR, head_color=tuple(min(255, x + y) for x, y in zip(SNAKE1_COLOR, SNAKE_HEAD_COLOR_OFFSET)))
        self.snake2 = Snake(initial_pos=(self.w - 6, self.h - 6), color=SNAKE2_COLOR, head_color=tuple(min(255, x + y) for x, y in zip(SNAKE2_COLOR, SNAKE_HEAD_COLOR_OFFSET)))

        # 餌と壁の配置
        self._place_food()
        self._place_walls()

        self.score1 = 0
        self.score2 = 0
        self.frame_iteration = 0

    def _move(self, snake, action):
        # action: [straight, right, left]
        # 現在の方向に基づいて、次の方向を決定
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(snake.direction)

        if action[1] == 1:  # 右折
            new_idx = (idx + 1) % 4
            snake.direction = clock_wise[new_idx]
        elif action[2] == 1:  # 左折
            new_idx = (idx - 1) % 4
            snake.direction = clock_wise[new_idx]
        # action[0] == 1 (直進) の場合は方向を変えない

        x, y = snake.head
        if snake.direction == Direction.RIGHT:
            x += 1
        elif snake.direction == Direction.LEFT:
            x -= 1
        elif snake.direction == Direction.DOWN:
            y += 1
        elif snake.direction == Direction.UP:
            y -= 1
        snake.head = Point(x, y)

    def _place_food(self):
        self.food = []
        while len(self.food) < FOOD_COUNT:
            x = random.randint(0, self.w - 1)
            y = random.randint(0, self.h - 1)
            point = Point(x, y)
            if point not in self.snake1.body and point not in self.snake2.body and point not in self.food:
                self.food.append(point)

    def _place_walls(self):
        self.walls = []
        while len(self.walls) < WALL_COUNT:
            x = random.randint(0, self.w - 1)
            y = random.randint(0, self.h - 1)
            point = Point(x, y)
            # 壁がスネークの初期位置やその周辺、餌の場所と重ならないようにする
            if (point not in self.snake1.body and point not in self.snake2.body and
                point not in self.food and point not in self.walls and
                # スネークのスタートエリアを確保
                not (4 <= x <= 6 and 4 <= y <= 6) and
                not (self.w - 7 <= x <= self.w - 5 and self.h - 7 <= y <= self.h - 5)):
                self.walls.append(point)

    def _place_new_food(self):
        while len(self.food) < FOOD_COUNT:
            x = random.randint(0, self.w - 1)
            y = random.randint(0, self.h - 1)
            point = Point(x, y)
            if point not in self.snake1.body and point not in self.snake2.body and point not in self.food and point not in self.walls:
                self.food.append(point)

    def play_step(self, action1, action2):
        self.frame_iteration += 1
        # 1. イベント処理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. スネークを動かす
        self._move(self.snake1, action1)
        self.snake1.body.insert(0, self.snake1.head)

        self._move(self.snake2, action2)
        self.snake2.body.insert(0, self.snake2.head)

        # 3. 衝突判定と報酬設定
        reward1 = 0
        reward2 = 0
        game_over = False

        # 餌を食べたか判定
        if self.snake1.head in self.food:
            self.score1 += 1
            reward1 = 0.3 # 餌を食べた報酬
            self.food.remove(self.snake1.head)
            self._place_new_food()
        else:
            self.snake1.body.pop()

        if self.snake2.head in self.food:
            self.score2 += 1
            reward2 = 0.3 # 餌を食べた報酬
            self.food.remove(self.snake2.head)
            self._place_new_food()
        else:
            self.snake2.body.pop()

        # 敗北条件の判定
        s1_dead = self._is_collision(self.snake1)
        s2_dead = self._is_collision(self.snake2)

        if s1_dead and s2_dead:
            reward1 = -0.2 # 引き分け
            reward2 = -0.2
            game_over = True
        elif s1_dead:
            reward1 = -1.0 # s1の負け
            reward2 = 1.0  # s2の勝ち
            game_over = True
        elif s2_dead:
            reward1 = 1.0  # s1の勝ち
            reward2 = -1.0 # s2の負け
            game_over = True
        
        reward1 -= 0.001 # ステップごとのペナルティ
        reward2 -= 0.001 # ステップごとのペナルティ
        

        reward1 += (self.snake1.length - self.snake2.length) * 0.002 # 長さの差による報酬
        reward2 += (self.snake2.length - self.snake1.length) * 0.002
        
        # 4. UI更新
        self._update_ui()
        self.clock.tick(20) # ゲーム速度

        # 4. UI更新
        self._update_ui()
        self.clock.tick(20) # ゲーム速度

        # 5. 戻り値
        return reward1, reward2, game_over, (self.score1, self.score2)

    def _is_collision(self, snake):
        # マップの端
        if not (0 <= snake.head.x < self.w and 0 <= snake.head.y < self.h):
            return True
        # 壁
        if snake.head in self.walls:
            return True
        # 自分自身
        if snake.head in snake.body[1:]:
            return True
        # 相手のスネーク
        other_snake = self.snake2 if snake == self.snake1 else self.snake1
        if snake.head in other_snake.body:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        # スネーク1を描画
        for i, pt in enumerate(self.snake1.body):
            color = self.snake1.head_color if i == 0 else self.snake1.color
            pygame.draw.rect(self.display, color, pygame.Rect(pt.x * GRID_SIZE, pt.y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

        # スネーク2を描画
        for i, pt in enumerate(self.snake2.body):
            color = self.snake2.head_color if i == 0 else self.snake2.color
            pygame.draw.rect(self.display, color, pygame.Rect(pt.x * GRID_SIZE, pt.y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

        # 餌を描画
        for pt in self.food:
            pygame.draw.rect(self.display, FOOD_COLOR, pygame.Rect(pt.x * GRID_SIZE, pt.y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

        # 壁を描画
        for pt in self.walls:
            pygame.draw.rect(self.display, WALL_COLOR, pygame.Rect(pt.x * GRID_SIZE, pt.y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

        # スコア表示
        text1 = font.render(f"Score 1: {self.score1}", True, WHITE)
        self.display.blit(text1, [10, 10])
        text2 = font.render(f"Score 2: {self.score2}", True, WHITE)
        self.display.blit(text2, [self.w * GRID_SIZE - text2.get_width() - 10, 10])

        pygame.display.flip()
