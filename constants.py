import torch

# --- ゲーム設定 ---
GRID_WIDTH = 30  # マップの幅
GRID_HEIGHT = 20  # マップの高さ
GRID_SIZE = 30  # 1マスのピクセルサイズ

SCREEN_WIDTH = GRID_WIDTH * GRID_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * GRID_SIZE

# --- 色 ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
GREEN = (0, 200, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)

# --- スネーク設定 ---
SNAKE1_COLOR = BLUE1
SNAKE2_COLOR = BLUE2
SNAKE_HEAD_COLOR_OFFSET = (0, 50, 50)  # 頭を少し明るくする

# --- 餌と壁 ---
FOOD_COLOR = RED
WALL_COLOR = GRAY
FOOD_COUNT = 3
WALL_COUNT = 10

# --- 強化学習 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
