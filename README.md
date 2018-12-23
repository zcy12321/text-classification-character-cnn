# text-classification-character-cnn
基于字符级别的textCNN中文文本分类

#*训练方法*
python run_cnn.py --help

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
