#!/usr/bin/python
# coding:utf8
"""
@author: Cong Yu
@time: 2019-04-02 20:02
"""
import os, time
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
from data_2 import get_input_data, load_dict, sequence_padding

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# 设置 gpu 显存使用量
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.65  # 占用GPU90%的显存


class RC:
    """
        text cnn + 模型的 tf 实现 relation classification
    """

    def __init__(self, vocab_size, num_classes, embedding_size=256, hidden_size=128, max_num=384, dropout=0.5,
                 l2_reg_lambda=0.0):
        # 参数
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.max_num = max_num
        # 输入 mask 只保留 sequence_length 的长度
        with tf.name_scope('placeholder'):
            # x的shape为[batch_size, 单词个数]
            # y的shape为[batch_size, num_classes]
            self.input_x = tf.placeholder(tf.int64, [None, max_num], name='input_x')
            self.input_x1 = tf.placeholder(tf.int64, [None, max_num], name='input_x1')
            self.input_x2 = tf.placeholder(tf.int64, [None, max_num], name='input_x2')
            self.output_predicate = tf.placeholder(tf.int64, [None, ], name='output_predicate')
            self.is_training = tf.placeholder(tf.bool, name='is_training')  # , name='is_training'
        # 区分 训练和测试
        l2_loss = tf.constant(0.0)
        if self.is_training is True:
            self.dropout = dropout
        else:
            self.dropout = 1.0
        # 构建模型
        # embedding 层 batch * max_sentence_num (?, 256, 64)
        with tf.name_scope("embedding"):
            # 从截断的正态分布输出随机值
            embedding_mat = tf.Variable(tf.truncated_normal((self.vocab_size, self.embedding_size)))
            # shape为[batch_size, max_num, embedding_size]
            input_x = tf.nn.embedding_lookup(embedding_mat, self.input_x)
            input_x1 = tf.nn.embedding_lookup(embedding_mat, self.input_x1)
            input_x2 = tf.nn.embedding_lookup(embedding_mat, self.input_x2)
            embedding_3 = tf.concat(
                (tf.expand_dims(input_x, 1), tf.expand_dims(input_x1, 1), tf.expand_dims(input_x2, 1)), 1)
        print("embedding_3", embedding_3.shape)

        with tf.name_scope("cnn_block"):
            cnn_1 = tf.squeeze(tf.layers.conv2d(embedding_3, self.hidden_size, (3, 2), activation=tf.nn.relu), [1])
            cnn_2 = tf.squeeze(tf.layers.conv2d(embedding_3, self.hidden_size, (3, 4), activation=tf.nn.relu), [1])
            cnn_3 = tf.squeeze(tf.layers.conv2d(embedding_3, self.hidden_size, (3, 8), activation=tf.nn.relu), [1])

            mx_1 = tf.reduce_max(cnn_1, 1)
            mx_2 = tf.reduce_max(cnn_2, 1)
            mx_3 = tf.reduce_max(cnn_3, 1)

            concat_vec = tf.concat((mx_1, mx_2, mx_3), 1)
        print("cnn_block", concat_vec.shape)

        with tf.name_scope("fc"):
            out_ = layers.fully_connected(inputs=concat_vec, num_outputs=self.hidden_size, activation_fn=tf.nn.relu)
        print("fc", out_.shape)

        with tf.name_scope("output"):
            W = tf.get_variable("W", shape=[self.hidden_size, num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(out_, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.predictions32 = tf.to_int32(self.predictions, name="predictions32")
        print("concat_vec", concat_vec.shape)
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,
                                                             labels=tf.one_hot(self.output_predicate, num_classes, 1,
                                                                               0))
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.output_predicate)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


def train():
    """

    :return:
    """
    char2id, schema2id = load_dict(char_dict="train_data/char2id.json", schema_dict="train_data/schema2id.json")
    # tf.flags.DEFINE_string("data_dir", "data/data.dat", "data directory")
    tf.flags.DEFINE_integer("vocab_size", len(char2id), "vocabulary size")
    tf.flags.DEFINE_integer("num_classes", len(schema2id), "number of classes")
    tf.flags.DEFINE_integer("max_num", 384, "max_sentence_num")
    tf.flags.DEFINE_integer("embedding_size", 256, "Dimensionality of character embedding (default: 200)")
    tf.flags.DEFINE_integer("hidden_size", 128, "Dimensionality of GRU hidden layer (default: 50)")
    tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 50)")
    tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
    tf.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 5)")
    tf.flags.DEFINE_integer("evaluate_every", 300, "evaluate every this many batches")
    tf.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
    tf.flags.DEFINE_float("grad_clip", 5, "grad clip to prevent gradient explode")
    FLAGS = tf.flags.FLAGS
    with tf.Session(config=config) as sess:
        rc = RC(vocab_size=FLAGS.vocab_size,
                num_classes=FLAGS.num_classes,
                embedding_size=FLAGS.embedding_size,
                hidden_size=FLAGS.hidden_size,
                max_num=FLAGS.max_num)

        # 外部定义 优化器
        global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        # RNN中常用的梯度截断，防止出现梯度过大难以求导的现象
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(rc.loss, tvars), FLAGS.grad_clip)
        grads_and_vars = tuple(zip(grads, tvars))
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
        if not os.path.exists('./ckpt_c/'):
            os.makedirs("./ckpt_c/")

        # 恢复模型 / 重新初始化参数
        # model_file = tf.train.latest_checkpoint('./ckpt/')
        ckpt = tf.train.get_checkpoint_state('./ckpt_c/')
        if ckpt:
            print("load saved model:\t", ckpt.model_checkpoint_path)
            saver = tf.train.Saver()
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("init model...")
            sess.run(tf.global_variables_initializer())

        def train_step(x, x1, x2, y):
            feed_dict = {
                rc.input_x: x,
                rc.input_x1: x1,
                rc.input_x2: x2,
                rc.output_predicate: y,
                rc.is_training: True,
            }
            _, step, cost, accuracy = sess.run(
                [train_op, global_step, rc.loss, rc.accuracy],
                feed_dict)
            time_str = str(int(time.time()))
            print("{}: step {}, loss {}, acc {}".format(time_str, step, cost, accuracy))
            # train_summary_writer.add_summary(summaries, step)
            return step

        def dev_step(x, x1, x2, y):
            feed_dict = {
                rc.input_x: x,
                rc.input_x1: x1,
                rc.input_x2: x2,
                rc.output_predicate: y,
                rc.is_training: True,
            }
            step, cost, accuracy = sess.run(
                [global_step, rc.loss, rc.accuracy],
                feed_dict)
            time_str = str(int(time.time()))
            length = len(y)
            print("+dev+{}: step {}, loss {}, t_acc {}".format(time_str, step, cost, accuracy))
            return cost, accuracy, length

        best_accuracy, best_at_step = 0, 0
        train_example_len = 349266
        dev_example_len = 43749
        num_train_steps = int(train_example_len / FLAGS.batch_size * FLAGS.num_epochs)
        num_dev_steps = int(dev_example_len / FLAGS.batch_size)
        max_acc = 0.0
        input_x_train, input_x1_train, input_x2_train, output_predicate_train = get_input_data(
            "./train_data/train_classification.tf_record", FLAGS.batch_size)
        input_x_dev, input_x1_dev, input_x2_dev, output_predicate_dev = get_input_data(
            "./train_data/dev_classification.tf_record", FLAGS.batch_size)
        # input_ids_train, output_types_train = get_input_data("./train_data/train_ner.tf_record", FLAGS.batch_size)
        # input_ids_dev, output_types_dev = get_input_data("./train_data/dev_ner.tf_record", FLAGS.batch_size)
        for i in range(num_train_steps):
            # batch 数据
            # input_ids_train_, output_types_train_ = sess.run([input_ids_train, output_types_train])
            input_x_train_, input_x1_train_, input_x2_train_, output_predicate_train_ = sess.run(
                [input_x_train, input_x1_train, input_x2_train, output_predicate_train])
            step = train_step(input_x_train_, input_x1_train_, input_x2_train_, output_predicate_train_)
            if step % FLAGS.evaluate_every == 0:
                # dev 数据过大， 也需要进行 分批
                total_dev_correct = 0
                total_devs = 0
                for j in range(num_dev_steps):
                    # input_ids_dev_, output_types_dev_ = sess.run([input_ids_dev, output_types_dev])
                    input_x_dev_, input_x1_dev_, input_x2_dev_, output_predicate_dev_ = sess.run(
                        [input_x_dev, input_x1_dev, input_x2_dev, output_predicate_dev])
                    loss, acc, count = dev_step(input_x_dev_, input_x1_dev_, input_x2_dev_,
                                                output_predicate_dev_)
                    total_dev_correct += count * acc
                    total_devs += count
                dev_accuracy = float(total_dev_correct) / total_devs
                print("预测：", total_dev_correct)
                print("长度为：", total_devs)
                print("最后预测结果：", dev_accuracy)
                if dev_accuracy > max_acc:
                    print("save model:\t%f\t>%f" % (dev_accuracy, max_acc))
                    max_acc = dev_accuracy
                    saver.save(sess, './ckpt_c/ner.ckpt', global_step=step)

        sess.close()


def predict(text, label1, label2):
    def load_model():
        output_graph_def = tf.GraphDef()

        with open('./ckpt_c/classification.pb', "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        return sess

    # 读入数据
    char2id, schema2id = load_dict(char_dict="train_data/char2id.json", schema_dict="train_data/schema2id.json")
    id2type = {value: key for key, value in schema2id.items()}
    ids = list(id2type.keys())
    sess = load_model()

    input_x = sess.graph.get_tensor_by_name("placeholder/input_x:0")
    input_x1 = sess.graph.get_tensor_by_name("placeholder/input_x1:0")
    input_x2 = sess.graph.get_tensor_by_name("placeholder/input_x2:0")
    # is_training = sess.graph.get_tensor_by_name("placeholder/Placeholder:0")  # is_training
    predict_y = sess.graph.get_tensor_by_name("output/scores:0")

    t1 = time.time()
    x = sequence_padding([char2id.get(c, 1) for c in text], max_len=384)
    subject = text.find(label1)
    temp = np.zeros(384, dtype=np.int64)
    temp[subject:subject + len(label1)] = x[subject:subject + len(label1)]
    x1 = temp
    object = text.find(label2)
    temp = np.zeros(384, dtype=np.int64)
    temp[object:object + len(label2)] = x[object:object + len(label2)]
    x2 = temp
    feed_dict = {
        input_x: [x],
        input_x1: [x1],
        input_x2: [x2],
    }
    predicts_d = sess.run([predict_y], feed_dict)[0]
    # soft max 求 1 vs n 概率
    p = np.exp(predicts_d[0]) / np.sum(np.exp(predicts_d[0]), axis=0)
    index = np.argmax(p)
    print(id2type[index], p[index])
    # 封装一下，输出结果


if __name__ == "__main__":
    1
    # with tf.Session() as sess:
    #     rc = RC(vocab_size=7025, num_classes=33, embedding_size=256, hidden_size=128, max_num=384)
    # train()
    text = "《猫喵》是晋江文学城连载的小说，作者是黑蓝色"
    label1 = "猫喵"
    label2 = "黑蓝色"
    predict(text, label1, label2)
