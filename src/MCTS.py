import numpy as np
import torch

# 使用Monte Carlo树搜索 (MCTS) 搜索策略
class TreeNode:
    def __init__(self, parent, prior):
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value_sum = 0
        self.prior = prior

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

    def get_value(self):
        if self.visits == 0:
            return 0
        return self.value_sum / self.visits

    def select(self, c_puct):
        best_value = -np.inf
        best_action = None
        best_child = None

        for action, child in self.children.items():
            u = c_puct * child.prior * np.sqrt(self.visits) / (1 + child.visits)
            q = child.get_value()

            value = q + u
            if value > best_value:
                best_value = value
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, prior_probs):
        for action, prob in np.ndenumerate(prior_probs):
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)

    def backup(self, value):
        if not self.is_root():
            self.parent.backup(-value)

        self.value_sum += value
        self.visits += 1


class MCTS:
    def __init__(self, net, c_puct=1.0, n_simulations=100):
        self.net = net
        self.c_puct = c_puct
        self.n_simulations = n_simulations

    def search(self, board):
        root = TreeNode(None, 1.0)

        for _ in range(self.n_simulations):
            node = root
            temp_board = board.copy()

            while not node.is_leaf():
                action, node = node.select(self.c_puct)
                temp_board.make_move(action)

            board_state = self.board_to_tensor(temp_board)
            with torch.no_grad():
                policy, value = self.net(torch.from_numpy(board_state).float().unsqueeze(0))

            policy = policy.squeeze(0).view(15, 15).numpy()
            valid_moves = (temp_board.board == 0)
            prior_probs = policy * valid_moves
            prior_probs /= np.sum(prior_probs)
            node.expand(prior_probs)

            value = value.item()
            node.backup(value)

        max_visits = -1
        best_move = None
        for action, child in root.children.items():
            if child.visits > max_visits:
                max_visits = child.visits
                best_move = action

        return best_move

    def board_to_tensor(self, board):
        board_state = np.zeros((3, 15, 15), dtype=np.float32)
        board_state[0] = (board.board == board.current_player)
        board_state[1] = (board.board == (3 - board.current_player))
        board_state[2] = (board.current_player == 1)
        return board_state