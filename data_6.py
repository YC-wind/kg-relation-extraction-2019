#!/usr/bin/python
# coding:utf8
"""
@author: Cong Yu
@time: 2019-04-11 22:20
"""
import json, collections
from tqdm import tqdm
import numpy as np
import tensorflow as tf

max_seq_len = 384
train_example_len = 173109
dev_example_len = 21639
batch_size = 256


# 添加 pos

def prepare_pos_dict():
    pos = """n	common nouns
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
    w	punctuations"""
    pos2id = dict()
    pos2id["<pad>"] = 0
    for index, p_ in enumerate(pos.split("\n")):
        pos2id[p_.split("\t")[0].strip()] = index + 1
    open("./train_data_6/pos2id.json", "w").write(json.dumps(pos2id, ensure_ascii=False, indent=2))


def prepare_schema2id():
    ner2id = json.loads(open("train_data_4/ner2id.json").read())
    schema2id = dict()
    schema2id["<pad>"] = 0
    schema2id["other"] = 1
    for k, v in ner2id.items():
        if v > 2:
            schema2id[k] = v - 1
    open("./train_data_6/schema2id.json", "w").write(json.dumps(schema2id, ensure_ascii=False, indent=2))


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


def prepare_data_position(path="./data/train_data.json", output="./train_data_6/train_position.tf_record"):
    """
        生成训练集 , 使用 IO 2-tag的方式
    :return:
    """
    # 加载 字典
    char2id, schema2id, pos2id = load_dict(char_dict="train_data_6/char2id.json",
                                           schema_dict="train_data_6/schema2id.json",
                                           pos_dict="train_data_6/pos2id.json")
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
            # 新增 pos
            input_pos = []
            for word in a["postag"]:
                input_pos += [pos2id.get(word["pos"])] * len(word["word"])
            input_pos = sequence_padding(input_pos, max_len=max_seq_len)
            # X.append(x)
            # 输出, 分两步走，第一步 定位 subject 的位置

            # 只取一个主体
            if len(a['spo_list']) < 1:
                continue
            # 定位主体
            s_text = a['spo_list'][0]["subject"]
            subject = text.find(a['spo_list'][0]["subject"])
            # 任务一，提取subject
            subject_tag = np.zeros(max_seq_len, dtype=np.int8)
            subject_tag[:len(text)] = 1  # pad 为0，其他other 为1
            subject_tag[subject:subject + len(s_text)] = 2  # 一共3个标签 0-pad 1-other 2-subject

            s1 = subject
            s2 = subject + len(s_text) - 1
            # 任务二，提取object
            object_tag = np.zeros(max_seq_len, dtype=np.int8)
            object_tag[:len(text)] = 1  # pad 为0，其他other 为1

            for sp in a['spo_list']:
                if sp["subject"] != s_text:
                    continue
                # 客体定位，并打上 predicate 标签
                ner_tag = "p_" + sp['predicate'] + "_s_" + sp['subject_type'] + "_o_" + sp['object_type']
                object = text.find(sp["object"])
                object_tag[object:object + len(sp["object"])] = schema2id.get(ner_tag)

            def create_int_feature(values):
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                return f

            features = collections.OrderedDict()
            features["input_chars"] = create_int_feature(x)
            features["input_pos"] = create_int_feature(input_pos)
            # subject 注意机制
            features["s1"] = create_int_feature([s1])
            features["s2"] = create_int_feature([s2])
            # subject object识别
            features["subject_tag"] = create_int_feature(subject_tag)
            features["object_tag"] = create_int_feature(object_tag)

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
            i += 1
    writer.close()
    print(i)


def get_input_data(input_file, batch_size):
    def parser(record):
        name_to_features = {
            "input_chars": tf.FixedLenFeature([max_seq_len], tf.int64),
            "input_pos": tf.FixedLenFeature([max_seq_len], tf.int64),

            "s1": tf.FixedLenFeature([], tf.int64),
            "s2": tf.FixedLenFeature([], tf.int64),
            "subject_tag": tf.FixedLenFeature([max_seq_len], tf.int64),
            "object_tag": tf.FixedLenFeature([max_seq_len], tf.int64),
        }

        example = tf.parse_single_example(record, features=name_to_features)
        input_ids = example["input_chars"]
        input_pos = example["input_pos"]

        s1 = example["s1"]
        s2 = example["s2"]
        subject_tag = example["subject_tag"]
        object_tag = example["object_tag"]

        return input_ids, input_pos, s1, s2, subject_tag, object_tag

    dataset = tf.data.TFRecordDataset(input_file)
    dataset = dataset.map(parser).repeat().batch(batch_size).shuffle(buffer_size=1000)
    iterator = dataset.make_one_shot_iterator()
    input_ids, input_pos, s1, s2, subject_tag, object_tag = iterator.get_next()
    return input_ids, input_pos, s1, s2, subject_tag, object_tag


def load_dict(char_dict="train_data_6/char2id.json", schema_dict="train_data_6/schema2id.json",
              pos_dict="train_data_6/pos2id.json"):
    """
        load dict
    :param char_dict:
    :param schema_dict:
    :param pos_dict:
    :return:
    """
    char2id = json.loads(open(char_dict).read())
    schema2id = json.loads(open(schema_dict).read())
    pos2id = json.loads(open(pos_dict).read())
    return char2id, schema2id, pos2id


if __name__ == "__main__":
    # generate_ner_tags_by_schema()
    # prepare_pos_dict()
    1
    # prepare_data_position(path="./data/train_data.json", output="./train_data_6/train_position.tf_record")
    # prepare_data_position(path="./data/dev_data.json", output="./train_data_6/dev_position.tf_record")
    input_ids_train, input_pos_train, s1_train, s2_train, subject_tag_train, object_tag_train = get_input_data(
        "./train_data_6/train_position.tf_record", 32)
    with tf.Session() as sess:
        input_ids_train_, input_pos_train_, s1_train_, s2_train_, subject_tag_train_, object_tag_train_ = sess.run(
            [input_ids_train, input_pos_train, s1_train, s2_train, subject_tag_train, object_tag_train])
        print(input_ids_train_.shape)
