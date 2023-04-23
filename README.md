# GobangChess_AI3.0
Starwing - 五子棋AI3.0版. 能够进行自我对弈并从中学习。

-------------------------------------------------------------------

2023.4.23

将运算设备从cpu改为显卡gpu

-------------------------------------------------------------------

2023.4.23

解决问题：在同一次训练中，每次自我对弈生的对局都是不变的。

原因：每次都从MCTS策略中选择最佳走法。

解决方法：修改了 generate_self_play_data 函数，使其在每次循环时，有一定概率（例如 5%）选择随机的合法走法，而不是从 MCTS 中选择最佳走法。

-------------------------------------------------------------------

2023.4.22

调整参数

修改保存部分代码，设计为在每一轮训练后都会自动保存模型至文件中。

-------------------------------------------------------------------

============================结构说明================================

定义棋盘表示及规则：Board 类表示棋盘，包含一些方法来进行落子、判断胜利和判断棋盘是否已满。

建立神经网络模型：NeuralNet 类继承自 PyTorch 的 nn.Module。它包含了卷积层、批量归一化层和激活层。此外，它还定义了策略头和价值头，用于输出落子概率和当前局面的估值。

使用 Monte Carlo 树搜索（MCTS）搜索策略：TreeNode 类表示蒙特卡罗树搜索中的一个节点。MCTS 类使用神经网络和棋盘状态进行蒙特卡罗树搜索，为每一步生成最佳的落子策略。

训练神经网络并生成自我对弈数据：generate_self_play_data 函数生成自我对弈数据，用于训练神经网络。train_neural_network 函数使用这些数据训练神经网络，优化策略和价值头的损失。

人机对战测试：human_vs_ai 函数实现了与AI的对战，人类玩家和AI轮流进行落子。在每一轮中，AI使用MCTS搜索最佳落子策略，而人类玩家输入他们的落子位置。

train.py文件：训练AI，参数自定义

test.py文件：人机对战测试

============================参数说明================================

在Board类中，棋盘表示为一个15x15的二维数组，用于存储棋子的位置。current_player表示当前玩家，初始值为1。

在NeuralNet类中，定义了一个卷积神经网络，包括三个卷积层、批标准化层和激活函数ReLU。网络的最后部分包括策略头和价值头，用于预测走子概率和局面价值。

在MCTS类的__init__方法中：

c_puct：默认值为1.0，是一个控制搜索过程中探索与利用平衡的超参数。

n_simulations：默认值为100，表示在搜索过程中要进行的模拟次数。

在generate_self_play_data函数中：

num_games：默认值为1000，表示自我对弈生成训练数据时，要进行的游戏数量。

在train_neural_network函数中：

batch_size：默认值为32，表示训练神经网络时的批量大小。

learning_rate：默认值为0.01，表示训练神经网络时的学习率。

epochs：默认值为1000，表示训练神经网络时的迭代次数。

这些参数可以根据实际需求进行调整，以达到更好的性能和效果。
