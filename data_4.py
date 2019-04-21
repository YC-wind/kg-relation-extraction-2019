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
    open("./train_data_4/pos2id.json", "w").write(json.dumps(pos2id, ensure_ascii=False, indent=2))


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


def prepare_data_ner(path="./data/train_data.json", output="./train_data_4/train_ner.tf_record"):
    """
        生成训练集 , 使用 IO 2-tag的方式
    :return:
    """
    # 加载 字典
    char2id = json.loads(open("train_data_4/char2id.json").read())
    ner2id = json.loads(open("train_data_4/ner2id.json").read())
    pos2id = json.loads(open("train_data_4/pos2id.json").read())
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

            # 输出
            y = np.zeros(max_seq_len, dtype=np.int8)
            y[:len(text)] = 1  # pad 为0，其他other 为1
            # 只取一个主体
            if len(a['spo_list']) < 1:
                continue
            # 定位主体
            s_text = a['spo_list'][0]["subject"]
            subject = text.find(a['spo_list'][0]["subject"])

            y[subject:subject + len(s_text)] = 2  # subject 下标为 2
            # subject_type_id = ner2id.get("subject_type_" + sp["subject_type"])

            for sp in a['spo_list']:
                if sp["subject"] != s_text:
                    continue
                # 客体定位，并打上 predicate 标签
                ner_tag = "p_" + sp['predicate'] + "_s_" + sp['subject_type'] + "_o_" + sp['object_type']
                object = text.find(sp["object"])
                y[object:object + len(sp["object"])] = ner2id.get(ner_tag)

            # Y.append(y)

            def create_int_feature(values):
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                return f

            features = collections.OrderedDict()
            features["input_chars"] = create_int_feature(x)
            features["input_pos"] = create_int_feature(input_pos)
            features["output"] = create_int_feature(y)
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
            "output": tf.FixedLenFeature([max_seq_len], tf.int64),
        }

        example = tf.parse_single_example(record, features=name_to_features)
        input_ids = example["input_chars"]
        input_pos = example["input_pos"]
        output_types = example["output"]
        return input_ids, input_pos, output_types

    dataset = tf.data.TFRecordDataset(input_file)
    dataset = dataset.map(parser).repeat().batch(batch_size).shuffle(buffer_size=1000)
    iterator = dataset.make_one_shot_iterator()
    input_ids, input_pos, output_types = iterator.get_next()
    return input_ids, input_pos, output_types


def load_dict(char_dict="train_data_4/char2id.json", ner_dict="train_data_4/ner2id.json",
              pos_dict="train_data_4/pos2id.json"):
    """
        load dict
    :param char_dict:
    :param ner_dict:
    :param pos_dict:
    :return:
    """
    char2id = json.loads(open(char_dict).read())
    ner2id = json.loads(open(ner_dict).read())
    pos2id = json.loads(open(pos_dict).read())
    return char2id, ner2id, pos2id


if __name__ == "__main__":
    # generate_ner_tags_by_schema()
    # prepare_pos_dict()
    1
    prepare_data_ner(path="./data/train_data.json", output="./train_data_4/train_ner.tf_record")
    prepare_data_ner(path="./data/dev_data.json", output="./train_data_4/dev_ner.tf_record")
