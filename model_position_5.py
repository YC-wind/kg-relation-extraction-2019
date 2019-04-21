#!/usr/bin/python
# coding:utf8
"""
@author: Cong Yu
@time: 2019-04-02 11:40
"""
import os, time, json
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
from data_5 import get_input_data, load_dict, sequence_padding

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# 设置 gpu 显存使用量
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.65  # 占用GPU90%的显存


def batch_norm_wrapper(inputs, is_training, decay=0.999):
    """
        bn1 = batch_norm_wrapper(z1, is_training)
        l1 = tf.nn.sigmoid(bn1)
    :param inputs:
    :param is_training:
    :param decay:
    :return:
    """
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs, [0])
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            # epsilon = 0.001
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, 0.001)
    else:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, 0.001)


def length(sequences):
    # 返回一个序列中每个元素的长度
    used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
    seq_len = tf.reduce_sum(used, reduction_indices=1)
    return tf.cast(seq_len, tf.int32)


class Position:
    """
        lstm + cnn + crf 模型的 tf 实现 ner
    """

    def __init__(self, vocab_size_c=5793, vocab_size_p=25, num_classes=53, embedding_size_c=256, embedding_size_p=256,
                 hidden_size=32, max_num=256, dropout=0.5):
        # 参数
        self.vocab_size_c = vocab_size_c
        self.vocab_size_p = vocab_size_p
        self.num_classes = num_classes
        self.embedding_size_c = embedding_size_c
        self.embedding_size_p = embedding_size_p
        self.hidden_size = hidden_size
        self.max_num = max_num
        # 输入 mask 只保留 sequence_length 的长度
        with tf.name_scope('placeholder'):
            # x的shape为[batch_size, 单词个数]
            # y的shape为[batch_size, num_classes]
            # 输入
            self.input_chars = tf.placeholder(tf.int32, [None, max_num], name='input_chars')
            self.input_pos = tf.placeholder(tf.int32, [None, max_num], name='input_pos')
            # 输出
            self.subject_start = tf.placeholder(tf.int32, [None, max_num], name='subject_start')
            self.subject_end = tf.placeholder(tf.int32, [None, max_num], name='subject_end')
            self.s1 = tf.placeholder(tf.int32, [None, ], name='s1')
            self.s2 = tf.placeholder(tf.int32, [None, ], name='s2')
            self.object_start = tf.placeholder(tf.int32, [None, max_num], name='object_start')
            self.object_end = tf.placeholder(tf.int32, [None, max_num], name='object_end')

            # self.output = tf.placeholder(tf.int32, [None, max_num], name='output')
            self.is_training = tf.placeholder(tf.bool, name='is_training')  # , name='is_training'
        subject_start = tf.cast(tf.expand_dims(self.subject_start, 2), tf.float32)
        subject_end = tf.cast(tf.expand_dims(self.subject_end, 2), tf.float32)

        # self.object_start = tf.cast(tf.one_hot(self.object_start, self.num_classes), tf.float32)
        # self.object_end = tf.cast(tf.one_hot(self.object_end, self.num_classes), tf.float32)

        self.input_m = tf.count_nonzero(self.input_chars, -1)  # 一维数据
        self.input_mask = tf.cast(tf.greater(tf.expand_dims(self.input_chars, 2), 0), tf.float32)
        print("input_mask", self.input_mask.shape)

        # 区分 训练和测试
        if self.is_training is True:
            self.dropout = 0.5
        else:
            self.dropout = 1.0
        # 构建模型
        # embedding 层 batch * max_sentence_num (?, 256, 64)
        word_embedded = self.word2vec()
        # 支持mask
        word_embedded = tf.multiply(word_embedded, self.input_mask)
        print("word_embedded", word_embedded.shape)

        pos_embedded = self.pos2vec()
        # 支持mask
        pos_embedded = tf.multiply(pos_embedded, self.input_mask)
        print("pos_embedded", pos_embedded.shape)

        # lstm 层 + dropout (?, 256, 128)
        vec_lstm_c = self.BidirectionalLSTMEncoder(word_embedded, name="bi-lstm-c")
        cnn_block_c = self.sent2vec_cnn(word_embedded, name="cnn_block_c")
        print("vec_lstm_c", vec_lstm_c.shape)
        print("cnn_block_c", cnn_block_c.shape)

        vec_lstm_p = self.BidirectionalLSTMEncoder(pos_embedded, name="bi-lstm-p")
        cnn_block_p = self.sent2vec_cnn(pos_embedded, name="cnn_block_p")
        print("vec_lstm_p", vec_lstm_p.shape)
        print("cnn_block_p", cnn_block_p.shape)

        # add_vec = tf.add_n([vec_lstm_c, vec_lstm_p, vec_lstm_p, cnn_block_p]) / 4
        # 可以求平均或者直接拼接
        self.con_vec = tf.concat((vec_lstm_c, vec_lstm_p, vec_lstm_p, cnn_block_p), 2)

        print("add_vec", self.con_vec.shape)

        # subject_start subject_end 预测
        self.ps1, self.ps2 = self.predict_subject()
        print("predict_subject", self.ps1.shape)

        # p o 预测
        self.po1, self.po2 = self.predict_po()
        print("predict_subject", self.ps1.shape)

        # loss
        s1_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=subject_start, logits=self.ps1)

        print("s1_loss", s1_loss.shape)
        s1_loss = tf.squeeze(s1_loss, axis=2)
        print("s1_loss", s1_loss.shape)
        input_mask = tf.squeeze(self.input_mask, axis=2)
        sum_loss = tf.reduce_sum(self.input_mask)
        print("sum_loss", sum_loss.shape)

        s1_loss = tf.reduce_sum(s1_loss * input_mask) / sum_loss
        s2_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=subject_end, logits=self.ps2)
        s2_loss = tf.squeeze(s2_loss)
        s2_loss = tf.reduce_sum(s2_loss * input_mask) / sum_loss
        # sparse_softmax_cross_entropy_with_logits 使用的时候，label不需要进行 one-hot
        o1_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.object_start, logits=self.po1)
        print("o1_loss", o1_loss.shape)
        o1_loss = tf.squeeze(o1_loss)
        o1_loss = tf.reduce_sum(o1_loss * input_mask) / sum_loss
        o2_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.object_end, logits=self.po2)
        o2_loss = tf.squeeze(o2_loss)
        o2_loss = tf.reduce_sum(o2_loss * input_mask) / sum_loss

        self.loss = 2.5 * (s1_loss + s2_loss) + (o1_loss + o2_loss)

    def word2vec(self):
        """
            # 嵌入层
        :return:
        """
        with tf.name_scope("embedding_c"):
            # 从截断的正态分布输出随机值
            embedding_mat = tf.Variable(tf.truncated_normal((self.vocab_size_c, self.embedding_size_c)))
            # shape为[batch_size, max_num, embedding_size]
            word_embedded = tf.nn.embedding_lookup(embedding_mat, self.input_chars)
        return word_embedded

    def pos2vec(self):
        """
            # 嵌入层
        :return:
        """
        with tf.name_scope("embedding_c"):
            # 从截断的正态分布输出随机值
            embedding_mat = tf.Variable(tf.truncated_normal((self.vocab_size_p, self.embedding_size_p)))
            # shape为[batch_size, max_num, embedding_size]
            pos_embedded = tf.nn.embedding_lookup(embedding_mat, self.input_pos)
        return pos_embedded

    def sent2vec_cnn(self, word_embedded, name="cnn-block"):
        """
            # GRU的输入tensor是[batch_size, max_time, ...].在构造句子向量时max_time应该是每个句子的长度，所以这里将
            *** # batch_size * sent_in_doc当做是batch_size.这样一来，每个GRU的cell处理的都是一个单词的词向量
            # 并最终将一句话中的所有单词的词向量融合（Attention）在一起形成句子向量
        :param word_embedded:
        :return:
        """
        with tf.name_scope(name):
            cnn_1 = tf.layers.conv1d(word_embedded, self.hidden_size * 2, 2, padding="same", activation=tf.nn.relu)
            # 添加 dropout 0.8
            dropout_1 = tf.nn.dropout(
                cnn_1,
                self.dropout,
                noise_shape=None,
                seed=None,
                name=None
            )
            cnn_2 = tf.layers.conv1d(word_embedded, self.hidden_size * 2, 3, padding="same", activation=tf.nn.relu)
            # 添加 dropout 0.8
            dropout_2 = tf.nn.dropout(
                cnn_2,
                self.dropout,
                noise_shape=None,
                seed=None,
                name=None
            )
            cnn_3 = tf.layers.conv1d(word_embedded, self.hidden_size * 2, 4, padding="same", activation=tf.nn.relu)
            # 添加 dropout 0.8
            dropout_3 = tf.nn.dropout(
                cnn_3,
                self.dropout,
                noise_shape=None,
                seed=None,
                name=None
            )
            # # 做法一：直接拼接
            # sent_vec = tf.concat((dropout_1, dropout_2, dropout_3), 2)
            # 做法二： 求平均
            sent_vec = tf.add_n([dropout_1, dropout_2, dropout_3]) / 3
            return sent_vec

    def BidirectionalLSTMEncoder(self, inputs, name):
        """
            # 双向GRU的编码层，将一句话中的所有单词或者一个文档中的所有句子向量进行编码得到一个 2×hidden_size的输出向量，
            然后在经过Attention层，将所有的单词或句子的输出向量加权得到一个最终的句子/文档向量。
            内部使用 GRU
        :param inputs:
        :param name:
        :return:
        """
        # 输入inputs的shape是 [batch_size, max_time, voc_size] = [batch_size * sent_in_doce, word_in_sent, embedding_size]
        with tf.variable_scope(name):
            LSTM_cell_fw = rnn.LSTMCell(self.hidden_size)
            LSTM_cell_bw = rnn.LSTMCell(self.hidden_size)
            # fw_outputs和bw_outputs的size都是[batch_size, max_time, hidden_size]
            ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=LSTM_cell_fw,
                                                                                 cell_bw=LSTM_cell_bw,
                                                                                 inputs=inputs,
                                                                                 sequence_length=self.input_m,
                                                                                 dtype=tf.float32)
            outputs = tf.concat((fw_outputs, bw_outputs), 2)
            return outputs

    def AttentionLayer(self, inputs, name):
        """
            然后Attention层就是一个MLP+softmax机制
            # inputs是GRU的输出，size是[batch_size, max_time, encoder_size(hidden_size * 2)] =
            # [batch_size * sent_in_doc, word_in_sent, hidden_size*2]
        :param inputs:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            u_context = tf.Variable(tf.truncated_normal([self.hidden_size * 2]), name='u_context')
            h = layers.fully_connected(inputs, self.hidden_size * 2, activation_fn=tf.nn.tanh)
            a = tf.reduce_sum(tf.multiply(h, u_context), axis=2, keepdims=True)
            alpha = tf.nn.softmax(a, axis=1)
            atten_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
            return atten_output

    def predict_subject(self):
        with tf.name_scope('predict_subject'):
            ps1 = tf.layers.dense(self.con_vec, 1, activation=tf.nn.sigmoid)
            ps2 = tf.layers.dense(self.con_vec, 1, activation=tf.nn.sigmoid)
            return ps1, ps2

    def predict_po(self):
        # 取出 s1 s2 对应的特征向量
        with tf.name_scope('predict_po'):
            batch_idxs = tf.expand_dims(tf.range(0, tf.shape(self.con_vec)[0]), 1)  # none,1
            print(self.s1.shape)

            s1 = tf.expand_dims(self.s1, 1)
            s2 = tf.expand_dims(self.s2, 1)

            id_s1 = tf.concat((batch_idxs, s1), 1)
            k1 = tf.gather_nd(self.con_vec, id_s1)
            id_s2 = tf.concat((batch_idxs, s2), 1)
            k2 = tf.gather_nd(self.con_vec, id_s2)
            kk = tf.concat((k1, k2), 1)  # b,2048

            vec = tf.expand_dims(kk, 1)  # b,1,2048
            vec = tf.zeros_like(self.con_vec[:, :, :1]) + vec
            fc = tf.concat((self.con_vec, vec), 2)

            # subject = tf.slice(self.con_vec, tf.expand_dims(self.s1, 2), tf.expand_dims(self.s1, 2))

            ps1 = tf.layers.dense(fc, self.num_classes)
            ps2 = tf.layers.dense(fc, self.num_classes)
            return ps1, ps2

    def classifer(self, doc_vec):
        """
            最后添加了一个 CRF 层，寻找最佳路径
        :param doc_vec:
        :return:
        """
        # 最终的输出层，是一个全连接层
        with tf.name_scope('doc_segment'):
            # 通过 改变shape ，实现 TimeDistributed
            # [batch_size*sent_in_doc, hidden_size*2]
            doc_vec_ = tf.reshape(doc_vec, [-1, doc_vec.shape[2]])
            # [batch_size*sent_in_doc, num_classes]
            # 中加在新增一层dense
            out_ = layers.fully_connected(inputs=doc_vec_, num_outputs=self.hidden_size, activation_fn=None)
            out_ = layers.fully_connected(inputs=out_, num_outputs=self.num_classes, activation_fn=None)
            # [batch_size, sent_in_doc, num_classes]
            out = tf.reshape(out_, [-1, self.max_num, self.num_classes])
            # out = layers.fully_connected(inputs=doc_vec, num_outputs=self.num_classes, activation_fn=None)
            # compute loss
            self.log_likelihood, self.transition_matrix = tf.contrib.crf.crf_log_likelihood(
                out, self.output, self.input_m)
            # inference
            self.viterbi_sequence, self.viterbi_score = tf.contrib.crf.crf_decode(
                out, self.transition_matrix, self.input_m)


def train():
    """
        模型训练
    :return:
    """
    char2id, schema2id, pos2id = load_dict(char_dict="train_data_5/char2id.json",
                                           schema_dict="train_data_5/schema2id.json",
                                           pos_dict="train_data_5/pos2id.json")
    # tf.flags.DEFINE_string("data_dir", "data/data.dat", "data directory")
    tf.flags.DEFINE_integer("vocab_size_c", len(char2id), "vocabulary size")
    tf.flags.DEFINE_integer("vocab_size_p", len(pos2id), "vocabulary size")
    tf.flags.DEFINE_integer("num_classes", len(schema2id), "number of classes")
    tf.flags.DEFINE_integer("max_num", 384, "max_sentence_num")
    tf.flags.DEFINE_integer("embedding_size_c", 256, "Dimensionality of character embedding (default: 200)")
    tf.flags.DEFINE_integer("embedding_size_p", 256, "Dimensionality of character embedding (default: 200)")
    tf.flags.DEFINE_integer("hidden_size", 128, "Dimensionality of GRU hidden layer (default: 50)")
    tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 50)")
    tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
    tf.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 5)")
    tf.flags.DEFINE_integer("evaluate_every", 300, "evaluate every this many batches")
    tf.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
    tf.flags.DEFINE_float("grad_clip", 5, "grad clip to prevent gradient explode")
    FLAGS = tf.flags.FLAGS
    with tf.Session(config=config) as sess:
        ner = Position(vocab_size_c=FLAGS.vocab_size_c,
                       vocab_size_p=FLAGS.vocab_size_p,
                       num_classes=FLAGS.num_classes,
                       embedding_size_c=FLAGS.embedding_size_c,
                       embedding_size_p=FLAGS.embedding_size_p,
                       hidden_size=FLAGS.hidden_size,
                       max_num=FLAGS.max_num)

        # 外部定义 优化器
        global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        # RNN中常用的梯度截断，防止出现梯度过大难以求导的现象
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(ner.loss, tvars), FLAGS.grad_clip)
        grads_and_vars = tuple(zip(grads, tvars))
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
        if not os.path.exists('./ckpt_5/'):
            os.makedirs("./ckpt_5/")

        # 恢复模型 / 重新初始化参数
        # model_file = tf.train.latest_checkpoint('./ckpt/')
        ckpt = tf.train.get_checkpoint_state('./ckpt_5/')
        if ckpt:
            print("load saved model:\t", ckpt.model_checkpoint_path)
            saver = tf.train.Saver()
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("init model...")
            sess.run(tf.global_variables_initializer())

        def evaluate(P, Y):
            '''
                计算变长的 准确率 指标
            :return:
            '''
            ps1, ps2, po1, po2 = P  # b,n,1    b,n,c
            po1, po2 = np.argmax(po1, 2), np.argmax(po2, 2)  # b,n
            subject_start, subject_end, object_start, object_end = Y

            def extract_subject(start, end, flag="p"):
                index = 0
                s1 = None
                s2 = None
                for s, e in zip(start, end):
                    if flag == "p":
                        if s[0] > 0.5:
                            s1 = index
                        if s1 is not None and e[0] > 0.5:
                            s2 = index
                            break
                    else:
                        if s > 0.5:
                            s1 = index
                        if s1 is not None and e > 0.5:
                            s2 = index
                            break
                    index += 1
                subject = (s1, s2)
                return subject

            def extract_object(start, end):  # n
                index = 0
                result = []
                s1 = None
                s2 = None
                tag = None
                for s, e in zip(start, end):
                    if s > 1:
                        s1 = index
                        tag = s
                    if s1 is not None and e == tag:
                        s2 = index
                        result.append((s1, s2, tag))
                        s1 = None
                        s2 = None
                        tag = None
                    index += 1
                return result

            A = 0
            B = 0
            C = 0
            for ps1_, ps2_, po1_, po2_, subject_start_, subject_end_, object_start_, object_end_ in zip(ps1, ps2, po1,
                                                                                                        po2,
                                                                                                        subject_start,
                                                                                                        subject_end,
                                                                                                        object_start,
                                                                                                        object_end):
                # 当前句子的长度
                p_subject = extract_subject(ps1_, ps2_)
                y_subject = extract_subject(subject_start_, subject_end_, flag="y")

                p_object = extract_object(po1_, po2_)
                y_object = extract_object(object_start_, object_end_)

                if p_subject == y_subject:
                    comm = [i for i in p_object if i in y_object]
                    A += len(comm)
                B += len(p_object)
                C += len(y_object)
            return A, B, C

        def train_step(x, pos, subject_start, subject_end, s1, s2, object_start, object_end):
            feed_dict = {
                ner.input_chars: x,
                ner.input_pos: pos,
                ner.subject_start: subject_start,
                ner.subject_end: subject_end,
                ner.s1: s1,
                ner.s2: s2,
                ner.object_start: object_start,
                ner.object_end: object_end,
                ner.is_training: True,
            }
            _, step, ps1, ps2, po1, po2, cost = sess.run(
                [train_op, global_step, ner.ps1, ner.ps2, ner.po1, ner.po2, ner.loss],
                feed_dict)
            tp, p_, r_ = evaluate((ps1, ps2, po1, po2), (subject_start, subject_end, object_start, object_end))
            time_str = str(int(time.time()))
            p = float(tp) / p_ if p_ else 0
            r = float(tp) / r_ if r_ else 0
            if p + r:
                f = 2 * p * r / (p + r)
            else:
                f = 0
            print("{}: step {}, loss {},  p {}, r {}, f {}".format(time_str, step, cost, p, r, f))
            # train_summary_writer.add_summary(summaries, step)
            return step

        def dev_step(x, pos, subject_start, subject_end, s1, s2, object_start, object_end):
            feed_dict = {
                ner.input_chars: x,
                ner.input_pos: pos,
                ner.subject_start: subject_start,
                ner.subject_end: subject_end,
                ner.s1: s1,
                ner.s2: s2,
                ner.object_start: object_start,
                ner.object_end: object_end,
                ner.is_training: False,
            }
            step, ps1, ps2, po1, po2, cost = sess.run(
                [global_step, ner.ps1, ner.ps2, ner.po1, ner.po2, ner.loss],
                feed_dict)

            tp, p_, r_ = evaluate((ps1, ps2, po1, po2), (subject_start, subject_end, object_start, object_end))

            time_str = str(int(time.time()))
            p = float(tp) / p_ if p_ else 0
            r = float(tp) / r_ if r_ else 0
            if p + r:
                f = 2 * p * r / (p + r)
            else:
                f = 0
            print("+dev+{}: step {}, loss {}, p {}, r {}, f {}".format(time_str, step, cost, p, r, f))
            # time_str = str(int(time.time()))
            # print("+dev+{}: step {}, loss {}, f_acc {}, t_acc {}".format(time_str, step, cost, accuracy, acc_d))
            return cost, tp, p_, r_

        best_accuracy, best_at_step = 0, 0

        train_example_len = 173109
        dev_example_len = 21639
        num_train_steps = int(train_example_len / FLAGS.batch_size * FLAGS.num_epochs)
        num_dev_steps = int(dev_example_len / FLAGS.batch_size)

        min_loss = 99999

        # input_ids_train, input_pos_train, output_types_train = get_input_data("./train_data_4/train_ner.tf_record",
        #                                                                       FLAGS.batch_size)
        input_ids_train, input_pos_train, subject_start_train, subject_end_train, s1_train, s2_train, object_start_train, object_end_train = get_input_data(
            "./train_data_5/train_position.tf_record", FLAGS.batch_size)
        # input_ids_dev, input_pos_dev, output_types_dev = get_input_data("./train_data_4/dev_ner.tf_record",
        #                                                                 FLAGS.batch_size)
        input_ids_dev, input_pos_dev, subject_start_dev, subject_end_dev, s1_dev, s2_dev, object_start_dev, object_end_dev = get_input_data(
            "./train_data_5/dev_position.tf_record", FLAGS.batch_size)
        for i in range(num_train_steps):
            # batch 数据
            # input_ids_train_, input_pos_train_, output_types_train_ = sess.run(
            #     [input_ids_train, input_pos_train, output_types_train])
            input_ids_train_, input_pos_train_, subject_start_train_, subject_end_train_, s1_train_, s2_train_, object_start_train_, object_end_train_ = sess.run(
                [input_ids_train, input_pos_train, subject_start_train, subject_end_train, s1_train, s2_train,
                 object_start_train, object_end_train])
            step = train_step(input_ids_train_, input_pos_train_, subject_start_train_, subject_end_train_, s1_train_,
                              s2_train_, object_start_train_, object_end_train_)
            if step % FLAGS.evaluate_every == 0:
                # dev 数据过大， 也需要进行 分批
                TP = 0
                P_ = 0
                R_ = 0
                total_loss = 0
                for j in range(num_dev_steps):
                    # input_ids_dev_, input_pos_dev_, output_types_dev_ = sess.run(
                    #     [input_ids_dev, input_pos_dev, output_types_dev])
                    input_ids_dev_, input_pos_dev_, subject_start_dev_, subject_end_dev_, s1_dev_, s2_dev_, object_start_dev_, object_end_dev_ = sess.run(
                        [input_ids_dev, input_pos_dev, subject_start_dev, subject_end_dev, s1_dev, s2_dev,
                         object_start_dev, object_end_dev])
                    loss, tp, p_, r_ = dev_step(input_ids_dev_, input_pos_dev_, subject_start_dev_, subject_end_dev_,
                                                s1_dev_, s2_dev_, object_start_dev_, object_end_dev_)
                    TP += tp
                    P_ += p_
                    R_ += r_
                    total_loss += loss
                    # total_dev_correct += count
                    # total_devs += total
                p = float(TP) / P_ if P_ else 0
                r = float(TP) / R_ if R_ else 0
                f = 2 * p * r / (p + r) if p + r else 0
                print("tp：p", TP, p)
                print("p_：r", P_, r)
                print("r_：f", R_, f)
                if total_loss < min_loss:
                    print("save model:\t%f\t>%f\t%f\t>%f" % (total_loss, p, r, f))
                    min_loss = total_loss
                    saver.save(sess, './ckpt_5/ner.ckpt', global_step=step)

        sess.close()


class Model:
    def __init__(self):
        output_graph_def = tf.GraphDef()

        with open('./ckpt_5/ner.pb', "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        # 读入数据
        self.char2id, self.ner2id, self.pos2id = load_dict(char_dict="train_data_4/char2id.json",
                                                           ner_dict="train_data_4/ner2id.json",
                                                           pos_dict="train_data_4/pos2id.json")
        self.id2type = {value: key for key, value in self.ner2id.items()}
        self.ids = list(self.id2type.keys())
        self.input_ids = self.sess.graph.get_tensor_by_name("placeholder/input_chars:0")
        self.input_pos = self.sess.graph.get_tensor_by_name("placeholder/input_pos:0")
        # is_training = sess.graph.get_tensor_by_name("placeholder/Placeholder:0")  # is_training
        self.viterbi_sequence = self.sess.graph.get_tensor_by_name("doc_segment/ReverseSequence_1:0")

    def predict(self, text, words):
        """

        :param text:
        :return:
        """
        # t1 = time.time()
        x = sequence_padding([self.char2id.get(c, 1) for c in text], max_len=384)
        input_pos = []
        for word in words:
            input_pos += [self.pos2id.get(word["pos"])] * len(word["word"])
        input_pos = sequence_padding(input_pos, max_len=384)

        feed_dict = {
            self.input_ids: [x],
            self.input_pos: [input_pos],
            # is_training: True,
        }
        predicts_d = self.sess.run([self.viterbi_sequence], feed_dict)[0]
        p = predicts_d.tolist()[0]
        # 封装一下，输出结果
        IOS = []
        index = 0
        start = None
        for i in p:
            if i == 0:
                if start is None:
                    pass
                else:
                    IOS.append((start, index))
                break
            elif i == 1:
                if start is None:
                    pass
                else:
                    if index > 0:
                        IOS.append((start, index))
                    start = None
            else:  # 包含实体
                if start is None:
                    start = index
                else:
                    if i == p[index - 1]:
                        pass
                    else:
                        IOS.append((start, index))
                        start = index
            index += 1
        # print(p)
        print(IOS)
        extract_dict = []
        # 首先找到 主题，即 为 2 的
        subject = ""
        for i in IOS:
            if p[i[0]] == 2:
                subject = text[i[0]:i[1]]
                break
        if subject != "":
            for i in IOS:
                if p[i[0]] > 2:
                    schema = schemas[p[i[0]] - 3]
                    object_ = text[i[0]:i[1]]
                    # {"predicate": "连载网站", "object_type": "网站", "subject_type": "网络小说", "object": "晋江文学城", "subject": "猫喵"}
                    schema["subject"] = subject
                    schema["object"] = object_
                    # extract_id = p[i[0]]
                    # tag = id2type.get(extract_id)
                    # value = text[i[0]:i[1]]
                    extract_dict.append(schema)
            if len(extract_dict) < 1:
                print(text)
        else:
            print(text)
        return extract_dict

    def submit(self):
        with open("result.json", "w") as fw:
            with open('./data/test1_data_postag.json') as fr:
                for l in (fr):
                    a = json.loads(l)
                    text = a["text"]
                    words = a["postag"]
                    spo_list = self.predict(text, words)
                    fw.write(json.dumps({"text": text, "spo_list": spo_list}, ensure_ascii=False))
                    fw.write("\n")


def predict(text):
    """

    :return:
    """

    def load_model():
        output_graph_def = tf.GraphDef()

        with open('./ckpt_5/ner.pb', "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        return sess

    # 读入数据
    char2id, ner2id = load_dict(char_dict="train_data_3/char2id.json", ner_dict="train_data_3/ner2id.json")
    id2type = {value: key for key, value in ner2id.items()}
    ids = list(id2type.keys())
    sess = load_model()

    input_ids = sess.graph.get_tensor_by_name("placeholder/input_chars:0")
    # is_training = sess.graph.get_tensor_by_name("placeholder/Placeholder:0")  # is_training
    viterbi_sequence = sess.graph.get_tensor_by_name("doc_segment/ReverseSequence_1:0")

    t1 = time.time()
    x = sequence_padding([char2id.get(c, 1) for c in text], max_len=384)
    feed_dict = {
        input_ids: [x],
        # is_training: True,
    }
    predicts_d = sess.run([viterbi_sequence], feed_dict)[0]
    p = predicts_d.tolist()[0]
    # 封装一下，输出结果
    IOS = []
    index = 0
    start = None
    for i in p:
        if i == 0:
            if start is None:
                pass
            else:
                IOS.append((start, index))
            break
        elif i == 1:
            if start is None:
                pass
            else:
                if index > 0:
                    IOS.append((start, index))
                start = None
        else:  # 包含实体
            if start is None:
                start = index
            else:
                if i == p[index - 1]:
                    pass
                else:
                    IOS.append((start, index))
                    start = index
        index += 1
    # print(p)
    print(IOS)
    extract_dict = []
    # 首先找到 主题，即 为 2 的
    subject = ""
    for i in IOS:
        if p[i[0]] == 2:
            subject = text[i[0]:i[1]]
            break
    if subject != "":
        for i in IOS:
            if p[i[0]] > 2:
                schema = schemas[p[i[0]] - 3]
                object_ = text[i[0]:i[1]]
                # {"predicate": "连载网站", "object_type": "网站", "subject_type": "网络小说", "object": "晋江文学城", "subject": "猫喵"}
                schema["subject"] = subject
                schema["object"] = object_
                # extract_id = p[i[0]]
                # tag = id2type.get(extract_id)
                # value = text[i[0]:i[1]]
                extract_dict.append(schema)
        if len(extract_dict) < 1:
            print(text)
    else:
        print(text)
    return extract_dict


def load_schemas():
    """

    :return:
    """
    schemas = []
    with open('./data/all_50_schemas') as f:
        for l in (f):
            a = json.loads(l)
            schemas.append(a)
    return schemas


if __name__ == "__main__":
    1
    # with tf.Session() as sess:
    #     ner = Position(vocab_size_c=7025, num_classes=33, embedding_size_c=256, hidden_size=128, max_num=384)
    #     tvars = tf.trainable_variables()
    #     opts = sess.graph.get_operations()
    #     for v in opts:
    #         print(v.name)
    #     for v in tvars:
    #         print(v.name)
    train()
    schemas = load_schemas()
    # m = Model()
    # m.submit()
    # spo_list = m.predict("作品赏析胡新良作品集胡新良作品中共邵阳县委党校胡新良，男，汉族，1968年7月生，湖南邵阳人",[{"word": "作品", "pos": "n"}, {"word": "赏析", "pos": "v"}, {"word": "胡新良", "pos": "nr"}, {"word": "作品集", "pos": "n"}, {"word": "胡新良", "pos": "nr"}, {"word": "作品", "pos": "n"}, {"word": "中共邵阳县委党校", "pos": "nt"}, {"word": "胡新良", "pos": "nr"}, {"word": "，", "pos": "w"}, {"word": "男", "pos": "a"}, {"word": "，", "pos": "w"}, {"word": "汉族", "pos": "nz"}, {"word": "，", "pos": "w"}, {"word": "1968年", "pos": "t"}, {"word": "7月", "pos": "t"}, {"word": "生", "pos": "v"}, {"word": "，", "pos": "w"}, {"word": "湖南", "pos": "ns"}, {"word": "邵阳", "pos": "ns"}, {"word": "人", "pos": "n"}])
    # print(spo_list)

    # spo_list = predict("《在温柔中疼痛》是由网络作家宋江创作的一部网络文学作品，发表于2008年，正文语种为简体中文，现已完成") # 测试结果 ok 的, -_-
    # print(spo_list)
