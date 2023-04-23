import numpy as np

# 定义棋盘表示及规则
class Board:
    def __init__(self):
        self.board = np.zeros((15, 15), dtype=int)
        self.current_player = 1

    def make_move(self, move):
        if self.board[move] == 0:
            self.board[move] = self.current_player
            self.current_player = 3 - self.current_player
            return True
        return False

    def random_move(self):
        valid_moves = np.argwhere(self.board == 0)
        random_move = valid_moves[np.random.choice(len(valid_moves))]
        return tuple(random_move)

    def is_win(self):
        for row in range(15):
            for col in range(15):
                if self.board[row][col] != 0:
                    player = self.board[row][col]
                    # 检查水平方向
                    if col <= 10 and all(self.board[row][col + i] == player for i in range(5)):
                        return True
                    # 检查竖直方向
                    if row <= 10 and all(self.board[row + i][col] == player for i in range(5)):
                        return True
                    # 检查右斜方向
                    if row <= 10 and col <= 10 and all(self.board[row + i][col + i] == player for i in range(5)):
                        return True
                    # 检查左斜方向
                    if row >= 4 and col <= 10 and all(self.board[row - i][col + i] == player for i in range(5)):
                        return True
        return False

    def is_full(self):
        return not np.any(self.board == 0)

    def copy(self):
        new_board = Board()
        new_board.board = np.copy(self.board)
        new_board.current_player = self.current_player
        return new_board