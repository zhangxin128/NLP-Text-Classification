#encoding:utf-8
from collections import Counter
#import tensorflow.contrib.keras as kr
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow import keras
import re
import numpy as np
import codecs

import jieba.analyse
#分好训练机测试集并进行jieba分词，u表示将后面跟的字符串以unicode格式存储
def read_file(filename):
    re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)")  # the method of cutting text by punctuation
    contents, labels = [], []
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                line = line.rstrip()
                assert len(line.split('\t')) == 2
                label, content = line.split('\t')
                labels.append(label)
                blocks = re_han.split(content)
                word = []
                for blk in blocks:
                    if re_han.match(blk):
                        word.extend(jieba.lcut(blk))
                contents.append(word)

            except:
                pass
    return labels, contents

def built_vocab_vector(filenames,voc_size = 10000):
    '''
    去停用词，得到前9999个词，获取对应的词 以及 词向量
    :param filenames:
    :param voc_size:
    :return:
    '''
    stopword = open('D:\THUCNEWS\stopwords.txt', 'r', encoding='utf-8')
    stop = [key.strip(' \n') for key in stopword]

    all_data = []
    j = 1
    embeddings = np.zeros([10000,100]) #建立1万行，100维的，里面全是0填充的矩阵

    for filename in filenames:#所有的txt文档都一遍去停用词
        labels, content = read_file(filename)
        for eachline in content:
            line =[]
            for i in range(len(eachline)):
                if str(eachline[i]) not in stop:#去停用词
                    line.append(eachline[i])
            all_data.extend(line)
            all_data = jieba.analyse.extract_tags(all_data, topK=10000, withWeight=False, allowPOS=())

    counter = Counter(all_data)#计数函数
    count_paris = counter.most_common(voc_size-1)#most_common()函数用来实现Top n 功能
    word, _ = list(zip(*count_paris))#zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这元些组组成的列表
#list() 方法用于将元组或字符串转换为列表。元组与列表是非常类似的，区别在于元组的元素值不能修改，元组是放在括号中，列表是放于方括号中。

    f = codecs.open('D:\THUCNEWS/vector_word.txt', 'r', encoding='utf-8')
    vocab_word = open('D:\THUCNEWS/cnews.vocab.txt', 'w', encoding='utf-8')
    for ealine in f:
        item = ealine.split(' ')
        key = item[0]#key的值是item这个数组里面的第一个。
        vec = np.array(item[1:], dtype='float32')
        if key in word:
            embeddings[j] = np.array(vec)
            vocab_word.write(key.strip('\r') + '\n')
            j += 1
    np.savez_compressed('D:\THUCNEWS/vector_word.npz', embeddings=embeddings)

def get_wordid(filename):#建立词典
    key = open(filename, 'r', encoding='utf-8')

    wordid = {}
    wordid['<PAD>'] = 0
    j = 1
    for w in key:
        w = w.strip('\n')
        w = w.strip('\r')
        wordid[w] = j
        j += 1
    return wordid


def read_category():#把标签也词典一下
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id

def process(filename, word_to_id, cat_to_id, max_length=100):
    labels, contents = read_file(filename)

    data_id, label_id = [], []
    #label_id = np.argmax(label_id, axis=1)
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    #label_id = np.argmax(label_id, axis=0)
    x_pad = keras.preprocessing.sequence.pad_sequences(data_id, max_length, padding='post', truncating='post')
    y_pad = keras.utils.to_categorical(label_id)
    return x_pad, y_pad

def get_word2vec(filename):
    with np.load(filename) as data:
        return data["embeddings"]


def batch_iter(x, y, batch_size =64):
    data_len = len(x)
    num_batch = int((data_len - 1)/batch_size) + 1
    indices = np.random.permutation(np.arange(data_len))#随机排列一个序列，或者数组
    '''
    np.arange(4) = [0,1,2,3]
    np.random.permutation([1, 4, 9, 12, 15]) = [15,  1,  9,  4, 12]
    '''
    x_shuff = x[indices]
    y_shuff = y[indices]
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i+1) * batch_size, data_len)#有时最后一个小batch可能不足64个，所以min（）
        yield x_shuff[start_id:end_id], y_shuff[start_id:end_id]

def seq_length(x_batch):#针对RNN的特征，计算了每个Batch的真实长度
    real_seq_len = []
    for line in x_batch:
        real_seq_len.append(np.sum(np.sign(line)))

    return real_seq_len






