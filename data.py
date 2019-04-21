#!/usr/bin/python
# coding:utf8
"""
@author: Cong Yu
@time: 2019-04-01 13:54
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


def count_spo_list():
    dev_data = []
    count_0 = 0
    count_1 = 0
    count_2 = 0
    count_3 = 0
    count_n = 0
    max_len = 0
    with open('./data/train_data.json') as f:
        for l in tqdm(f):
            a = json.loads(l)
            dev_data.append(
                {
                    'text': a['text'],
                    'spo_list': [(i['subject'], i['predicate'], i['object']) for i in a['spo_list']]
                }
            )
            if len(a["text"]) > max_len:
                max_len = len(a["text"])
            if len(a["spo_list"]) == 0:
                count_0 += 1
            elif len(a["spo_list"]) == 1:
                count_1 += 1
            elif len(a["spo_list"]) == 2:
                count_2 += 1
            elif len(a["spo_list"]) == 3:
                count_3 += 1
            else:
                count_n += 1
    print(count_0, count_1, count_2, count_3, count_n)


def generate_char_count():
    """
        生成 字频
    :return:
    """
    char_count = dict()
    # char2id = dict()
    # char2id["<pad>"] = 0
    # char2id["<unk>"] = 1
    with open('./data/train_data.json') as f:
        for l in tqdm(f):
            a = json.loads(l)
            for c in a['text']:
                # char2id[c] = char2id.get(c, 0) + 1
                if char_count.get(c):
                    char_count[c] += 1
                else:
                    char_count[c] = 1
    with open('./data/dev_data.json') as f:
        for l in tqdm(f):
            a = json.loads(l)
            for c in a['text']:
                # char2id[c] = char2id.get(c, 0) + 1
                if char_count.get(c):
                    char_count[c] += 1
                else:
                    char_count[c] = 1
    sorted_x = sorted(char_count.items(), key=lambda x: x[1], reverse=True)
    open("./train_data/char_count.json", "w").write(json.dumps(sorted_x, ensure_ascii=False))


def filter_char_dict(min_count=5):
    """
        按词频过滤 char
    :return:
    """
    char_count = json.loads(open("./train_data/char_count.json").read())
    # char_count.reverse()
    print(len(char_count))
    char2id = dict()
    char2id["<pad>"] = 0
    char2id["<unk>"] = 1
    index_c = 2
    for item in char_count:
        if item[1] >= min_count:  # 可以优化，后面不用比较啦
            char2id[item[0]] = index_c
            index_c += 1
    print(len(char2id))
    open("./train_data/char2id.json", "w").write(json.dumps(char2id, ensure_ascii=False, indent=2))


def parser_schema():
    """
        解析 schema的 类型
    :return:
    """
    dict_schemas = set()
    dict_types = set()
    with open('./data/all_50_schemas') as f:
        for l in tqdm(f):
            a = json.loads(l)
            dict_schemas.add(a['predicate'])
            dict_types.add("subject_type_" + a['subject_type'])
            dict_types.add("object_type_" + a['object_type'])

    dict_schemas = list(sorted(list(dict_schemas), key=lambda x: x, reverse=True))
    dict_schemas.insert(0, "other")
    id2predicate = {i: j for i, j in enumerate(dict_schemas)}  # 0表示终止类别
    predicate2id = {j: i for i, j in id2predicate.items()}
    open("./train_data/schema2id.json", "w").write(json.dumps(predicate2id, ensure_ascii=False, indent=2))

    dict_types = list(sorted(list(dict_types), key=lambda x: x, reverse=True))
    dict_types.insert(0, "other")
    dict_types.insert(0, "<pad>")
    id2type = {i: j for i, j in enumerate(dict_types)}  # 0表示终止类别
    type2id = {j: i for i, j in id2type.items()}
    open("./train_data/type2id.json", "w").write(json.dumps(type2id, ensure_ascii=False, indent=2))


def prepare_data_ner(path="./data/train_data.json", output="./train_data/train_ner.tf_record"):
    """
        生成训练集 , 使用 IO 2-tag的方式
        173109
        21639
    :return:
    """
    # 加载 字典
    char2id = json.loads(open("train_data/char2id.json").read())
    type2id = json.loads(open("train_data/type2id.json").read())
    # X = []
    # Y = []
    writer = tf.python_io.TFRecordWriter(output)
    with open(path) as f:
        i = 0
        for l in tqdm(f):
            i += 1
            a = json.loads(l)
            # 输入
            text = a['text']
            x = sequence_padding([char2id.get(c, 1) for c in text], max_len=max_seq_len)
            # X.append(x)
            # 输出
            y = np.zeros(max_seq_len, dtype=np.int8)
            y[:len(text)] = 1 # pad 为0，其他other 为1
            for sp in a['spo_list']:
                subject = text.find(sp["subject"])
                subject_type_id = type2id.get("subject_type_" + sp["subject_type"])
                y[subject:subject + len(sp["subject"])] = subject_type_id
                object = text.find(sp["object"])
                object_type_id = type2id.get("object_type_" + sp["object_type"])
                y[object:object + len(sp["object"])] = object_type_id

            # Y.append(y)

            def create_int_feature(values):
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                return f

            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(x)
            features["output_types"] = create_int_feature(y)
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
    writer.close()
    print(i)


def get_input_data(input_file, batch_size):
    def parser(record):
        name_to_features = {
            "input_ids": tf.FixedLenFeature([max_seq_len], tf.int64),
            "output_types": tf.FixedLenFeature([max_seq_len], tf.int64),
        }

        example = tf.parse_single_example(record, features=name_to_features)
        input_ids = example["input_ids"]
        output_types = example["output_types"]
        return input_ids, output_types

    dataset = tf.data.TFRecordDataset(input_file)
    dataset = dataset.map(parser).repeat().batch(batch_size).shuffle(buffer_size=1000)
    iterator = dataset.make_one_shot_iterator()
    input_ids, output_types = iterator.get_next()
    return input_ids, output_types


def load_dict(char_dict="train_data/char2id.json", type_dict="train_data/type2id.json"):
    """
        load dict
    :param char_dict:
    :param type_dict:
    :return:
    """
    char2id = json.loads(open(char_dict).read())
    type2id = json.loads(open(type_dict).read())
    return char2id, type2id


if __name__ == "__main__":
    # count_spo_list()
    # generate_char_count()
    # filter_char_dict()
    # parser_schema()
    # prepare_data_ner(path="./data/train_data.json", output="./train_data/train_ner.tf_record")
    # prepare_data_ner(path="./data/dev_data.json", output="./train_data/dev_ner.tf_record")
    1
    # input_ids_train, output_types_train = get_input_data("./train_data/train_ner.tf_record",batch_size)
    # with tf.Session() as sess:
    #     input_ids_train, output_types_train = sess.run([input_ids_train, output_types_train])
    #     print(input_ids_train.shape)
