#!/usr/bin/python
# coding:utf8
"""
@author: Cong Yu
@time: 2019-04-02 19:04
"""
import json, collections
from tqdm import tqdm
import numpy as np
import tensorflow as tf

max_seq_len = 384
train_example_len = 349266
dev_example_len = 43749
batch_size = 256

pos = """
POS	Meaning
n	common nouns
f	localizer
s	space
t	time
nr	noun of people
ns	noun of space
nt	noun of tuan
nw	noun of work
nz	other proper noun
v	verbs
vd	verb of adverbs
vn	verb of noun
a	adjective
ad	adjective of adverb
an	adnoun
d	adverbs
m	numeral
q	quantity
r	pronoun
p	prepositions
c	conjunction
u	auxiliary
xc	other function word
w	punctuations
"""


def sequence_padding(chars, padding="right", max_len=512):
    """
        对句子进行padding
    :return:
    """
    # list的extend方法没有返回值，是none，结果在原列表中
    l = len(chars)
    if padding == "left":
        if l <= max_len:
            _chars = [0] * (max_len - l) + chars
            # _labels = [0] * (max_len - l) + labels
            # _masks = [0] * (max_len - l) + [1] * l
        else:
            _chars = chars[l - max_len:]
            # _labels = labels[l - max_len:]
            # _masks = [1] * max_len
    elif padding == "right":
        if l <= max_len:
            _chars = chars + [0] * (max_len - l)
            # _labels = labels + [0] * (max_len - l)
            # _masks = [1] * l + [0] * (max_len - l)
        else:
            _chars = chars[:max_len]
            # _labels = labels[:max_len]
            # _masks = [1] * max_len
    else:
        raise Exception
    return _chars  # , _labels  # , _masks


def prepare_data_classification(path="./data/train_data.json", output="./train_data/train_classification.tf_record"):
    """
        生成训练集 , 使用 IO 2-tag的方式
        173109
        21639
    :return:
    """
    # 加载 字典
    char2id = json.loads(open("train_data/char2id.json").read())
    # type2id = json.loads(open("train_data/type2id.json").read())
    schema2id = json.loads(open("train_data/schema2id.json").read())

    def create_int_feature(values):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return f

    # X = []
    # Y = []
    writer = tf.python_io.TFRecordWriter(output)
    with open(path) as f:
        i = 0
        for l in tqdm(f):

            a = json.loads(l)
            # 输入
            text = a['text']
            x = sequence_padding([char2id.get(c, 1) for c in text], max_len=max_seq_len)
            # X.append(x)
            for sp in a['spo_list']:
                subject = text.find(sp["subject"])
                temp = np.zeros(max_seq_len, dtype=np.int64)
                temp[subject:subject + len(sp["subject"])] = x[subject:subject + len(sp["subject"])]
                x1 = temp
                object = text.find(sp["object"])
                temp = np.zeros(max_seq_len, dtype=np.int64)
                temp[object:object + len(sp["object"])] = x[object:object + len(sp["object"])]
                x2 = temp
                # y
                predicate = schema2id.get(sp["predicate"], 0)
                y = predicate

                # Y.append(y)
                i += 1
                features = collections.OrderedDict()
                features["input_x"] = create_int_feature(x)
                features["input_x1"] = create_int_feature(x1)
                features["input_x2"] = create_int_feature(x2)
                features["output_predicate"] = create_int_feature([y])
                tf_example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(tf_example.SerializeToString())
    writer.close()
    print(i)


def get_input_data(input_file, batch_size):
    def parser(record):
        name_to_features = {
            "input_x": tf.FixedLenFeature([max_seq_len], tf.int64),
            "input_x1": tf.FixedLenFeature([max_seq_len], tf.int64),
            "input_x2": tf.FixedLenFeature([max_seq_len], tf.int64),
            "output_predicate": tf.FixedLenFeature([], tf.int64),
        }

        example = tf.parse_single_example(record, features=name_to_features)
        input_x = example["input_x"]
        input_x1 = example["input_x1"]
        input_x2 = example["input_x2"]
        output_predicate = example["output_predicate"]
        return input_x, input_x1, input_x2, output_predicate

    dataset = tf.data.TFRecordDataset(input_file)
    dataset = dataset.map(parser).repeat().batch(batch_size).shuffle(buffer_size=1000)
    iterator = dataset.make_one_shot_iterator()
    input_x, input_x1, input_x2, output_predicate = iterator.get_next()
    return input_x, input_x1, input_x2, output_predicate


def load_dict(char_dict="train_data/char2id.json", schema_dict="train_data/schema2id.json"):
    """
        load dict
    :param char_dict:
    :param type_dict:
    :return:
    """
    char2id = json.loads(open(char_dict).read())
    schema2id = json.loads(open(schema_dict).read())
    return char2id, schema2id


if __name__ == "__main__":
    1
    prepare_data_classification(path="./data/train_data.json", output="./train_data/train_classification.tf_record")
    prepare_data_classification(path="./data/dev_data.json", output="./train_data/dev_classification.tf_record")
    # input_x_train, input_x1_train, input_x2_train, output_predicate_train = get_input_data(
    #     "./train_data/train_classification.tf_record", batch_size)
    # with tf.Session() as sess:
    #     input_x_train_, input_x1_train_, input_x2_train_, output_predicate_train_ = sess.run(
    #         [input_x_train, input_x1_train, input_x2_train, output_predicate_train])
    #     print(input_x_train_.shape)
