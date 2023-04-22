import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

from src.Board import Board
from src.MCTS import MCTS
from src.NeuralNet import NeuralNet


# 训练神经网络并生成自我对弈数据
def generate_self_play_data(net, mcts, num_games=50):
    self_play_data = []

    for game in range(num_games):
        # 显示每轮游戏的进度
        print(f"Game {game + 1}/{num_games}")
        game_data = []
        board = Board()

        while not board.is_win() and not board.is_full():
            move = mcts.search(board)
            print(f"Game {game + 1}/{num_games} - move: {move}")
            board_state = mcts.board_to_tensor(board)
            game_data.append((board_state, move))
            board.make_move(move)

        winner = 0 if board.is_full() else board.current_player
        for board_state, move in game_data:
            value = 1 if winner == 0 else (1 if board_state[2, 0, 0] == (winner == 1) else -1)
            self_play_data.append((board_state, move, value))

    return self_play_data


def train_neural_network(net, self_play_data, batch_size=32, learning_rate=0.01, epochs=20):
    print(self_play_data)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        random.shuffle(self_play_data)
        for i in range(0, len(self_play_data), batch_size):
            current_batch_size = min(batch_size, len(self_play_data) - i)
            batch = self_play_data[i:i + current_batch_size]
            board_states, moves, values = zip(*batch)

            board_states = torch.from_numpy(np.stack(board_states)).float()
            moves = torch.tensor([move[0] * 15 + move[1] for _, move, _ in batch]).long()[:current_batch_size]
            values = torch.tensor(values, dtype=torch.float).unsqueeze(-1)

            optimizer.zero_grad()

            policy_logits, pred_values = net(board_states)
            policy_logits = policy_logits.view(-1, 15 * 15)[:current_batch_size]

            policy_loss = nn.CrossEntropyLoss()(policy_logits, moves)
            value_loss = loss_fn(pred_values, values)
            loss = policy_loss + value_loss

            # 显示每个批次的损失
            print(
                f"Batch {i // batch_size + 1}/{len(self_play_data) // batch_size}: Policy Loss = {policy_loss.item():.4f}, Value Loss = {value_loss.item():.4f}")

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs} completed.")


def save_model(net, file_path="model.pth"):
    torch.save(net.state_dict(), file_path)
    print(f"Model saved to {file_path}")


def load_model(file_path="model.pth"):
    net = NeuralNet()
    if os.path.exists(file_path):
        net.load_state_dict(torch.load(file_path))
        net.eval()
        print(f"Model loaded from {file_path}")
    else:
        print(f"Model file {file_path} not found.")
    return net


def main():
    # 初始化神经网络，MCTS

    # 如果存在保存的模型，则加载它，否则训练一个新模型
    if os.path.exists("model.pth"):
        net = load_model()
    else:
        net = NeuralNet()

    mcts = MCTS(net)
    iterations = 20
    for iteration in range(iterations):
        # 显示每次迭代的进度
        print(f"Iteration {iteration + 1}/{iterations}")
        # 生成自我对弈数据
        self_play_data = generate_self_play_data(net, mcts)
        # 训练神经网络
        train_neural_network(net, self_play_data)

        # 保存训练好的模型
        # 改动-每跑一轮就保存一次
        save_model(net)


if __name__ == "__main__":
    main()
