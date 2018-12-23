# text-classification-character-cnn
基于字符级别的textCNN中文文本分类

# **训练方法**
python run_cnn.py --help
可查看参数设置，均为可选参数，默认值在run_cnn.py中
默认执行trian
```
usage: run_cnn.py [-h] [--mode MODE] [--train_dir TRAIN_DIR]
                  [--val_dir VAL_DIR] [--test_dir TEST_DIR]
                  [--vocab_dir VOCAB_DIR] [--save_dir SAVE_DIR]
                  [--save_path SAVE_PATH]

命令行参数设置

optional arguments:
  -h, --help            show this help message and exit
  --mode MODE           train or test
  --train_dir TRAIN_DIR
                        训练数据文件路径
  --val_dir VAL_DIR     验证数据文件路径
  --test_dir TEST_DIR   测试数据文件路径
  --vocab_dir VOCAB_DIR
                        词汇表文件路径
  --save_dir SAVE_DIR   最佳验证结果保存文件夹
  --save_path SAVE_PATH
                        最佳验证结果保存路径
```
# **验证方法**
python run_cnn.py --mode=test
```
Testing...
Test Loss:   0.17, Test Acc:  95.53%
Precision, Recall and F1-Score...
                 precision    recall  f1-score   support

         3C数码配件       0.96      0.90      0.93       200
    汽车/用品/配件/改装       0.95      0.99      0.97       200
模玩/动漫/周边/cos/桌游       0.96      0.99      0.98       200
          五金/工具       0.93      0.93      0.93       200
           家装主材       0.97      0.97      0.97       200
            餐饮具       0.99      0.99      0.99       200
  服饰配件/皮带/帽子/围巾       1.00      0.99      0.99       200
             手表       0.98      0.98      0.98       200
  饰品/流行首饰/时尚饰品新       0.99      0.97      0.98       200
        电子元器件市场       0.87      0.86      0.87       200
   办公设备/耗材/相关服务       0.97      0.96      0.97       200
           生活电器       0.94      0.90      0.92       200
    乐器/吉他/钢琴/配件       0.98      0.98      0.98       200
           厨房电器       0.92      0.95      0.94       200
           影音电器       0.91      0.94      0.93       200
            大家电       0.95      0.96      0.95       200

      micro avg       0.96      0.96      0.96      3200
      macro avg       0.96      0.96      0.96      3200
   weighted avg       0.96      0.96      0.96      3200

Confusion Matrix...
[[180   2   1   0   0   2   0   0   0   5   2   0   0   1   5   2]
 [  0 198   1   0   1   0   0   0   0   0   0   0   0   0   0   0]
 [  1   0 199   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   1   0 185   0   0   0   0   0   6   0   0   0   0   7   1]
 [  0   0   1   2 195   0   0   0   1   0   0   1   0   0   0   0]
 [  0   1   1   0   0 198   0   0   0   0   0   0   0   0   0   0]
 [  0   0   1   0   0   0 198   0   0   0   0   0   0   0   1   0]
 [  0   0   2   2   0   0   0 196   0   0   0   0   0   0   0   0]
 [  1   0   1   0   0   0   0   3 195   0   0   0   0   0   0   0]
 [  3   5   0   6   0   1   0   0   0 173   1   0   3   6   1   1]
 [  1   0   0   1   0   0   0   0   0   3 192   0   0   0   1   2]
 [  0   1   0   0   4   0   0   0   0   1   0 180   0   9   0   5]
 [  0   0   0   0   0   0   0   0   0   0   0   0 197   0   3   0]
 [  0   0   0   2   0   0   0   0   0   1   0   7   0 190   0   0]
 [  1   0   0   1   0   0   0   0   0   8   0   0   1   0 189   0]
 [  0   0   0   1   1   0   0   0   0   1   2   3   0   0   0 192]]
Time usage: 0:00:03
```

# **预测方法**
python predict.py --help
```
usage: predict.py [-h] [--predict_sentence PREDICT_SENTENCE]
                  [--predict_dir PREDICT_DIR]
                  [--predict_save_dir PREDICT_SAVE_DIR]
                  [--vocab_dir VOCAB_DIR] [--save_dir SAVE_DIR]
                  [--save_path SAVE_PATH]

命令行参数设置

optional arguments:
  -h, --help            show this help message and exit
  --predict_sentence PREDICT_SENTENCE
                        预测一个句子
  --predict_dir PREDICT_DIR
                        预测数据文件路径
  --predict_save_dir PREDICT_SAVE_DIR
                        预测数据文件保存路径
  --vocab_dir VOCAB_DIR
                        词汇表文件路径
  --save_dir SAVE_DIR   最佳验证结果保存文件夹
  --save_path SAVE_PATH
                        最佳验证结果保存路径
```
例如：python predict.py --predict_sentence='凡亚比 OTG转接头Type-c转USB适用小米6华为P9乐视数据线手机U盘连接转换头器
'
```
3C数码配件
```
预测整个txt文件：
python predict.py
```
预测结束 文件位置： data/products/products.predict_result.csv
```
