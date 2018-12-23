# text-classification-character-cnn
基于字符级别的textCNN中文文本分类
CNN可配置的参数如下所示，在cnn_model.py中
class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 64      # 词向量维度
    seq_length = 600        # 序列长度
    num_classes = 10        # 类别数
    num_filters = 128        # 卷积核数目
    kernel_size = 5         # 卷积核尺寸
    vocab_size = 5000       # 词汇表达小

    hidden_dim = 128        # 全连接层神经元

    dropout_keep_prob = 0.5 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 64         # 每批训练大小
    num_epochs = 10         # 总迭代轮次

    print_per_batch = 10    # 每多少轮输出一次结果
    save_per_batch = 10      # 每多少轮存入tensorboard
    
    CNN模型
具体参看cnn_model.py的实现。

大致结构如下：

images/cnn_architecture
