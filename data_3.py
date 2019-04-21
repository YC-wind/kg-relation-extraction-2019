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


def generate_ner_tags_by_schema():
    """
        解析 schema的 类型
    :return:
    """
    ner2id = dict()
    ner2id["<pad>"] = 0
    ner2id["other"] = 1
    ner2id["subject"] = 2
    index = 3
    with open('./data/all_50_schemas') as f:
        for l in tqdm(f):
            a = json.loads(l)
            ner2id["p_" + a['predicate'] + "_s_" + a['subject_type'] + "_o_" + a['object_type']] = index
            index += 1
    open("./train_data_3/ner2id.json", "w").write(json.dumps(ner2id, ensure_ascii=False, indent=2))


def prepare_data_ner(path="./data/train_data.json", output="./train_data_3/train_ner.tf_record"):
    """
        生成训练集 , 使用 IO 2-tag的方式
    :return:
    """
    # 加载 字典
    char2id = json.loads(open("train_data_3/char2id.json").read())
    ner2id = json.loads(open("train_data_3/ner2id.json").read())
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
            "output": tf.FixedLenFeature([max_seq_len], tf.int64),
        }

        example = tf.parse_single_example(record, features=name_to_features)
        input_ids = example["input_chars"]
        output_types = example["output"]
        return input_ids, output_types

    dataset = tf.data.TFRecordDataset(input_file)
    dataset = dataset.map(parser).repeat().batch(batch_size).shuffle(buffer_size=1000)
    iterator = dataset.make_one_shot_iterator()
    input_ids, output_types = iterator.get_next()
    return input_ids, output_types


def load_dict(char_dict="train_data_3/char2id.json", ner_dict="train_data_3/ner2id.json"):
    """
        load dict
    :param char_dict:
    :param ner_dict:
    :return:
    """
    char2id = json.loads(open(char_dict).read())
    ner2id = json.loads(open(ner_dict).read())
    return char2id, ner2id


if __name__ == "__main__":
    # generate_ner_tags_by_schema()
    1
    # prepare_data_ner(path="./data/train_data.json", output="./train_data_3/train_ner.tf_record")
    # prepare_data_ner(path="./data/dev_data.json", output="./train_data_3/dev_ner.tf_record")
