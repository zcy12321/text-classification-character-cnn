#coding:utf-8

import pandas as pd
from collections import Counter
import numpy as np
import tensorflow.contrib.keras as kr

#train/val/test数据制造，从excel文件中提取
origin_filename = 'products\描述性标记_cd.xlsx'

def split_train_eval_test(filename):
    df = pd.read_excel(filename, sheet_name='总表—含类目')
    # 统计顶级目录中的类目数
    categories = dict(df['顶级类目'].value_counts())
    categories_up500 = []
    for k,v in categories.items():
        if v>500:
            categories_up500.append(k)
    df.loc[df['顶级类目'].isin(categories_up500)][['宝贝标题','顶级类目']].to_csv('products\products_all_data.csv', encoding='utf-8')
    df_ls = []
    for i in range(len(categories_up500)):
        df_ls.append(df.loc[df['顶级类目']==categories_up500[i]].sample(n=500))
    #16个类目，每个类目500条随机标题，进行组合,总共8000条
    df_concat = pd.concat(df_ls)
    #train.txt(400*16)
    with open('products\products.train.txt', 'w', encoding='utf-8') as fw:
        for i in range(len(categories_up500)):
            df = df_concat.loc[df_concat['顶级类目'] == categories_up500[i]].iloc[:400]
            for _ in df['宝贝标题']:
                fw.write(categories_up500[i]+'\t'+_+'\n')
    #eval.txt(100*16),eval与train不重复
    with open('products\products.val.txt', 'w', encoding='utf-8') as fw:
        for i in range(len(categories_up500)):
            df = df_concat.loc[df_concat['顶级类目'] == categories_up500[i]].iloc[-100:]
            for _ in df['宝贝标题']:
                fw.write(categories_up500[i]+'\t'+_+'\n')
    #test.txt(200*16),test是在500中随机采样的
    with open('products\products.test.txt', 'w', encoding='utf-8') as fw:
        for i in range(len(categories_up500)):
            df = df_concat.loc[df_concat['顶级类目'] == categories_up500[i]].sample(n=200)
            for _ in df['宝贝标题']:
                fw.write(categories_up500[i]+'\t'+_+'\n')
    #predict.txt
    with open('products\products.predict.txt', 'w', encoding='utf-8') as fw:
        df = pd.read_excel(filename, sheet_name='总表—含类目')
        for _ in df.loc[df['顶级类目'] == '3C数码配件'].sample(n=10000)['宝贝标题']:
            fw.write(_+'\n')
        # for _ in df_concat.sample(n=3000)['宝贝标题']:
        #     fw.write(_ + '\n')


#数据加载函数
def open_file(filename, mode='r'):
    """
    mode: 'r' or 'w' for read or write
    """
    return open(filename, mode, encoding='utf-8', errors='ignore')


def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(content))
                    labels.append(label)
            except:
                pass
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储在vocab_dir"""
    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    with open_file(vocab_dir, mode='w') as fw:
        fw.write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """读取词汇表"""
    with open_file(vocab_dir) as fr:
        # 如果是py2 则每个值都转化为unicode
        words = [_.strip() for _ in fr.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():
    """读取分类目录，固定"""
    categories = ['3C数码配件', '汽车/用品/配件/改装', '模玩/动漫/周边/cos/桌游', '五金/工具', '家装主材', '餐饮具', '服饰配件/皮带/帽子/围巾', '手表', '饰品/流行首饰/时尚饰品新', '电子元器件市场', '办公设备/耗材/相关服务', '生活电器', '乐器/吉他/钢琴/配件', '厨房电器', '影音电器', '大家电']
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

if __name__=='__main__':
    split_train_eval_test(origin_filename)
