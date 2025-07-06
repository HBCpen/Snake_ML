from agent import Agent
from game import SnakeGame
from constants import *
import torch
import numpy as np
from model import CombinedNet, QTrainer

def train():
    # プロット用の変数を初期化
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    # エージェントとゲーム環境を初期化
    agent1 = Agent()
    agent2 = Agent()
    game = SnakeGame()

    # モデルの入力サイズを正しく設定
    # グリッドデータのチャンネル数と、ベクトルデータの長さを取得
    grid_state_ex, vector_state_ex = agent1.get_state(game, 1)
    num_channels = grid_state_ex.shape[1]
    vector_len = vector_state_ex.shape[1]

    agent1.model = CombinedNet(num_channels, vector_len, 3).to(DEVICE)
    agent1.trainer = QTrainer(agent1.model, lr=0.001, gamma=0.9)
    agent2.model = CombinedNet(num_channels, vector_len, 3).to(DEVICE)
    agent2.trainer = QTrainer(agent2.model, lr=0.001, gamma=0.9)

    while True:
        # 現在の状態を取得
        state_old1_grid, state_old1_vector = agent1.get_state(game, 1)
        state_old2_grid, state_old2_vector = agent2.get_state(game, 2)

        # 餌への距離を計算 (報酬シェイピング用)
        dist_before1 = np.inf
        if game.food:
            dist_before1 = min(np.linalg.norm(np.array(game.snake1.head) - np.array(f)) for f in game.food)
        dist_before2 = np.inf
        if game.food:
            dist_before2 = min(np.linalg.norm(np.array(game.snake2.head) - np.array(f)) for f in game.food)

        # 行動を取得
        final_move1 = agent1.get_action((state_old1_grid, state_old1_vector))
        final_move2 = agent2.get_action((state_old2_grid, state_old2_vector))

        # ゲームを1ステップ進める
        reward1, reward2, done, scores = game.play_step(final_move1, final_move2)

        # 時間経過ペナルティ
        reward1 -= 0.001
        reward2 -= 0.001

        # 距離変化による報酬シェイピング
        dist_after1 = np.inf
        if game.food:
            dist_after1 = min(np.linalg.norm(np.array(game.snake1.head) - np.array(f)) for f in game.food)
        reward1 += (dist_before1 - dist_after1) * 0.005

        dist_after2 = np.inf
        if game.food:
            dist_after2 = min(np.linalg.norm(np.array(game.snake2.head) - np.array(f)) for f in game.food)
        reward2 += (dist_before2 - dist_after2) * 0.005

        state_new1_grid, state_new1_vector = agent1.get_state(game, 1)
        state_new2_grid, state_new2_vector = agent2.get_state(game, 2)

        # 短期記憶で学習
        agent1.train_short_memory((state_old1_grid, state_old1_vector), final_move1, reward1, (state_new1_grid, state_new1_vector), done)
        agent2.train_short_memory((state_old2_grid, state_old2_vector), final_move2, reward2, (state_new2_grid, state_new2_vector), done)

        # 経験を記憶
        agent1.remember((state_old1_grid, state_old1_vector), final_move1, reward1, (state_new1_grid, state_new1_vector), done)
        agent2.remember((state_old2_grid, state_old2_vector), final_move2, reward2, (state_new2_grid, state_new2_vector), done)

        if done:
            # ゲームリセット
            game.reset()
            agent1.n_games += 1
            agent2.n_games += 1

            # 長期記憶で学習 (経験再生)
            agent1.train_long_memory()
            agent2.train_long_memory()

            # モデルの保存
            if scores[0] > record: # agent1のスコアで記録更新
                record = scores[0]
                agent1.model.save(file_name='model_agent1.pth')
            # agent2.model.save(file_name='model_agent2.pth') # 必要に応じて

            print('Game', agent1.n_games, 'Score', scores, 'Record:', record)

            # TODO: プロット処理

if __name__ == '__main__':
    train()
