# coding: utf-8

import argparse
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.keras as kr

from cnn_model import TCNNConfig, TextCNN
from data.data_helper import read_category, read_vocab

#命令行参数设置
parser = argparse.ArgumentParser(description='命令行参数设置')

parser.add_argument('--predict_sentence', default=None,type=str,help='预测一个句子')
parser.add_argument('--predict_dir',default='data/products/products.predict.txt',type=str,help='预测数据文件路径')
parser.add_argument('--predict_save_dir', default='data/products/products.predict_result.csv', type=str, help='预测数据文件保存路径')
parser.add_argument('--vocab_dir',default='data/products/products.vocab.txt',type=str,help='词汇表文件路径')

parser.add_argument('--save_dir',default='checkpoints/products_textcnn',type=str,help='最佳验证结果保存文件夹')
parser.add_argument('--save_path',default='checkpoints/products_textcnn/best_validation',help='最佳验证结果保存路径')
args = parser.parse_args()

class CnnModel:
    def __init__(self):
        self.config = TCNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(args.vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextCNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=args.save_path)  # 读取保存的模型

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        data = [self.word_to_id[x] for x in message if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        return self.categories[y_pred_cls[0]]


if __name__ == '__main__':
    cnn_model = CnnModel()
    if args.predict_sentence is not None:
        print(cnn_model.predict(args.predict_sentence))
    predict_result= []
    with open(args.predict_dir, 'r', encoding='utf-8') as fr:
        contents = fr.readlines()
        for _ in contents:
            predict_result.append(cnn_model.predict(_.strip()))
    df = pd.DataFrame({'宝贝标题':contents,'顶级类目':predict_result})
    df.to_csv(args.predict_save_dir, index=False, encoding='gbk')
    print('预测结束','文件位置：',args.predict_save_dir)

    # test_demo = ['索尼镜头盖40.5mm微单A5000A5100A6000A6300 NEX5 6相机16-50配件',
    #              '适配男式运动鞋垫男女异克软鞋垫子杀菌耐克鞋垫学生男鞋吸汗棉鞋',
    #              '科勒原装正品智能马桶盖板配件 智能盖板电源线线夹配件包',
    #              '简约气压式玻璃瓶二合一茅台酒居家手动开酒器啤酒创意便携式罐头',
    #              '高达模型打磨抛光工具绒面极细目打磨布文玩琥珀双面抛光布研磨布'
    #              ]
    # for i in test_demo:
    #     print(cnn_model.predict(i))
