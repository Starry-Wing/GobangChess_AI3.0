# GobangChess_AI3.0
Starwing - 五子棋AI3.0版. 具有学习能力

定义棋盘表示及规则：Board 类表示棋盘，包含一些方法来进行落子、判断胜利和判断棋盘是否已满。

建立神经网络模型：NeuralNet 类继承自 PyTorch 的 nn.Module。它包含了卷积层、批量归一化层和激活层。此外，它还定义了策略头和价值头，用于输出落子概率和当前局面的估值。

使用 Monte Carlo 树搜索（MCTS）搜索策略：TreeNode 类表示蒙特卡罗树搜索中的一个节点。MCTS 类使用神经网络和棋盘状态进行蒙特卡罗树搜索，为每一步生成最佳的落子策略。

训练神经网络并生成自我对弈数据：generate_self_play_data 函数生成自我对弈数据，用于训练神经网络。train_neural_network 函数使用这些数据训练神经网络，优化策略和价值头的损失。

人机对战测试：human_vs_ai 函数实现了与AI的对战，人类玩家和AI轮流进行落子。在每一轮中，AI使用MCTS搜索最佳落子策略，而人类玩家输入他们的落子位置。

train.py文件：训练AI，参数自定义

test.py文件：人机对战测试
