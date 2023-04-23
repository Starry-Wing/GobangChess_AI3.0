import numpy as np
import torch

# 运算设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


# net：神经网络对象，用于估计给定棋盘状态下的策略（走子概率）和价值（预测的结果）。这个神经网络对象是从NeuralNet类创建的。
#
# c_puct：默认值为1.0，是一个控制搜索过程中探索与利用平衡的超参数。c_puct越大，MCTS越注重探索未知的走子，越小则更倾向于重复访问高价值的走子。这个参数在MCTS树搜索过程中的select方法中被用来计算UCB（Upper Confidence Bound）值，进而影响选择哪一个子节点。
#
# n_simulations：默认值为100，表示在搜索过程中要进行的模拟次数。每一次模拟都从根节点开始，沿着树进行选择、扩展、模拟和回传步骤，直到达到终止条件。模拟次数越多，MCTS能够更充分地搜索可能的走子，但计算量也会相应增加。

class MCTS:
    def __init__(self, net, c_puct=1.2, n_simulations=80):
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

            board_state = torch.from_numpy(self.board_to_tensor(temp_board)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                policy, value = self.net(board_state)

            policy = policy.squeeze(0).view(15, 15).cpu().numpy()
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