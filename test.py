import torch
import random
import os

from src.Board import Board
from src.MCTS import MCTS
from src.NeuralNet import NeuralNet


def load_model(file_path="model.pth"):
    net = NeuralNet()
    if os.path.exists(file_path):
        net.load_state_dict(torch.load(file_path))
        net.eval()
        print(f"Model loaded from {file_path}")
    else:
        print(f"Model file {file_path} not found.")
    return net


# 人机对战测试
def human_vs_ai(net, mcts):
    board = Board()
    player = random.choice([1, 2])
    print(f"Human player is {player}, AI player is {3 - player}")

    while not board.is_win() and not board.is_full():
        if board.current_player == player:
            move = None
            while move is None:
                try:
                    row, col = map(int, input("Enter your move (row, col): ").split())
                    if 0 <= row < 15 and 0 <= col < 15 and board.board[row][col] == 0:
                        move = (row, col)
                    else:
                        print("Invalid move. Please try again.")
                except ValueError:
                    print("Invalid input. Please try again.")
        else:
            move = mcts.search(board)
            print(f"AI move: {move}")

        board.make_move(move)
        print(board.board)

    if board.is_win():
        print(f"Player {board.current_player} wins!")
    else:
        print("It's a draw!")


def main():
    # 与AI对战
    if os.path.exists("model.pth"):
        net = load_model()
        mcts = MCTS(net)
        while True:
            human_vs_ai(net, mcts)
            play_again = input("Do you want to play again? (y/n): ")
            if play_again.lower() != 'y':
                break
    else:
        print("no model")


if __name__ == "__main__":
    main()
