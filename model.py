import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

from constants import *

class CombinedNet(nn.Module):
    def __init__(self, num_channels, vector_len, num_actions):
        super(CombinedNet, self).__init__()
        # CNN for grid data
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # CNNの出力サイズを計算
        conv_output_size = self._get_conv_output_size(num_channels, GRID_WIDTH, GRID_HEIGHT)

        # Fully connected layers
        self.fc1_combined = nn.Linear(conv_output_size + vector_len, 256)
        self.fc2_output = nn.Linear(256, num_actions)

    def _get_conv_output_size(self, num_channels, w, h):
        with torch.no_grad():
            x = torch.zeros(1, num_channels, w, h)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            return x.flatten().shape[0]

    def forward(self, grid_data, vector_data):
        # CNN path
        x_grid = self.pool(F.relu(self.conv1(grid_data)))
        x_grid = self.pool(F.relu(self.conv2(x_grid)))
        x_grid = x_grid.view(x_grid.size(0), -1) # Flatten

        # Combine
        x_combined = torch.cat((x_grid, vector_data), dim=1)

        # Fully connected layers
        x = F.relu(self.fc1_combined(x_combined))
        x = self.fc2_output(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # データをテンソルに変換 (バッチ処理対応)
        # state[0]がタプルならバッチ処理、テンソルなら単一処理
        if isinstance(state[0], tuple):
            state_grid = torch.cat([s[0] for s in state], dim=0)
            state_vector = torch.cat([s[1] for s in state], dim=0)
        else: # short memory (バッチサイズ1)
            state_grid, state_vector = state

        action = torch.tensor(action, dtype=torch.long, device=DEVICE)
        reward = torch.tensor(reward, dtype=torch.float, device=DEVICE)

        if isinstance(next_state[0], tuple):
            next_state_grid = torch.cat([s[0] for s in next_state], dim=0)
            next_state_vector = torch.cat([s[1] for s in next_state], dim=0)
        else: # short memory
            next_state_grid, next_state_vector = next_state

        if len(action.shape) == 1: # short memory
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: 予測されたQ値 (現在の状態)
        pred = self.model(state_grid, state_vector)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                next_pred = self.model(next_state_grid[idx].unsqueeze(0), next_state_vector[idx].unsqueeze(0))
                Q_new = reward[idx] + self.gamma * torch.max(next_pred)

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 2: Q_new と pred の間の損失を計算
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
