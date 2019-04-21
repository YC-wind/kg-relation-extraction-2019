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
from data_6 import get_input_data, load_dict, sequence_padding

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
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

    def __init__(self, vocab_size_c=5793, vocab_size_p=25, num_classes=53, embedding_size_c=256, embedding_size_p=32,
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
            self.s1 = tf.placeholder(tf.int32, [None, ], name='s1')
            self.s2 = tf.placeholder(tf.int32, [None, ], name='s2')
            self.subject_tag = tf.placeholder(tf.int32, [None, max_num], name='subject_tag')
            self.object_tag = tf.placeholder(tf.int32, [None, max_num], name='object_tag')

            # self.output = tf.placeholder(tf.int32, [None, max_num], name='output')
            self.is_training = tf.placeholder(tf.bool, name='is_training')  # , name='is_training'

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
        pos_embedded = self.pos2vec()
        embedded = tf.concat((word_embedded, pos_embedded), 2)
        embedded = tf.multiply(embedded, self.input_mask)
        print("embedded", embedded.shape)

        vec_lstm = self.BidirectionalLSTMEncoder(embedded, name="bi-lstm-c")
        cnn_block = self.sent2vec_cnn(word_embedded, name="cnn_block_c")
        print("vec_lstm", vec_lstm.shape)
        print("cnn_block", cnn_block.shape)

        # 可以求平均或者直接拼接
        self.con_vec = tf.concat((vec_lstm, cnn_block), 2)
        print("add_vec", self.con_vec.shape)

        # 先 p o 预测
        self.loss_o, self.p_o, self.p_o_feature = self.predict_po()
        print(self.p_o.shape)
        print(self.p_o_feature.shape)

        # 再 subject 预测
        self.logits, self.p_s = self.predict_subject()
        input_mask = tf.squeeze(self.input_mask, axis=2)
        sum_loss = tf.reduce_sum(self.input_mask)
        print(self.p_s.shape)
        loss_s = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.subject_tag, logits=self.logits)
        loss_s = tf.squeeze(loss_s)
        # 不计算 mask 的loss
        self.loss_s = tf.reduce_sum(loss_s * input_mask) / sum_loss

        # loss_s 0.06621704250574112, loss_o 1.8898606300354004, 可以看出 loss 很不 平衡
        # 主要原因是， subject 只需识别一个 shiti， loss 相对较少，很多 1的计算 loss 会产生偏差，因此需要调整
        self.loss = 100.0 * self.loss_s + self.loss_o

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
            print(self.p_o_feature.shape)  # n,m,52
            cc = tf.concat((self.con_vec, self.p_o_feature, self.p_o_feature), 2)
            print(cc.shape)
            doc_vec_ = tf.reshape(cc, [-1, cc.shape[2]])
            # [batch_size*sent_in_doc, num_classes]
            # 中加在新增一层dense
            out_ = layers.fully_connected(inputs=doc_vec_, num_outputs=self.hidden_size, activation_fn=tf.nn.relu)
            out_ = layers.fully_connected(inputs=out_, num_outputs=3, activation_fn=None)
            print("out_", out_.shape)
            ps = tf.nn.softmax(out_, axis=-1)
            print("ps", ps.shape)
            ps = tf.argmax(tf.reshape(ps, [-1, self.max_num, 3]), 2, name="subject")
            # [batch_size, sent_in_doc, num_classes]
            out = tf.reshape(out_, [-1, self.max_num, 3])
            # 避免 2个 crf 冲突
            return out, ps

    def predict_po(self):
        # 取出 s1 s2 对应的特征向量
        """
        tf crf 使用两个会报错
         if transition_params is None:
            transition_params = vs.get_variable("transitions", [num_tags, num_tags])
        :return:
        """
        with tf.name_scope('predict_po'):
            doc_vec_ = tf.reshape(self.con_vec, [-1, self.con_vec.shape[2]])
            # [batch_size*sent_in_doc, num_classes]
            # 中加在新增一层dense
            out_ = layers.fully_connected(inputs=doc_vec_, num_outputs=self.hidden_size, activation_fn=None)
            out_ = layers.fully_connected(inputs=out_, num_outputs=self.num_classes, activation_fn=None)
            # [batch_size, sent_in_doc, num_classes]
            out = tf.reshape(out_, [-1, self.max_num, self.num_classes])
            # out = layers.fully_connected(inputs=doc_vec, num_outputs=self.num_classes, activation_fn=None)
            # compute loss
            log_likelihood_o, transition_matrix_o = tf.contrib.crf.crf_log_likelihood(
                out, self.object_tag, self.input_m)
            # inference
            viterbi_sequence_o, viterbi_score_o = tf.contrib.crf.crf_decode(
                out, transition_matrix_o, self.input_m)
            # 定义损失
            loss_o = -tf.reduce_mean(log_likelihood_o)
            return loss_o, viterbi_sequence_o, out


def train():
    """
        模型训练
    :return:
    """
    char2id, schema2id, pos2id = load_dict(char_dict="train_data_6/char2id.json",
                                           schema_dict="train_data_6/schema2id.json",
                                           pos_dict="train_data_6/pos2id.json")
    # tf.flags.DEFINE_string("data_dir", "data/data.dat", "data directory")
    tf.flags.DEFINE_integer("vocab_size_c", len(char2id), "vocabulary size")
    tf.flags.DEFINE_integer("vocab_size_p", len(pos2id), "vocabulary size")
    tf.flags.DEFINE_integer("num_classes", len(schema2id), "number of classes")
    tf.flags.DEFINE_integer("max_num", 384, "max_sentence_num")
    tf.flags.DEFINE_integer("embedding_size_c", 256, "Dimensionality of character embedding (default: 200)")
    tf.flags.DEFINE_integer("embedding_size_p", 32, "Dimensionality of character embedding (default: 200)")
    tf.flags.DEFINE_integer("hidden_size", 128, "Dimensionality of GRU hidden layer (default: 50)")
    tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", 3, "Number of training epochs (default: 50)")
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
        if not os.path.exists('./ckpt_7/'):
            os.makedirs("./ckpt_7/")

        # 恢复模型 / 重新初始化参数
        # model_file = tf.train.latest_checkpoint('./ckpt/')
        ckpt = tf.train.get_checkpoint_state('./ckpt_7/')
        if ckpt:
            print("load saved model:\t", ckpt.model_checkpoint_path)
            saver = tf.train.Saver()
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("init model...")
            sess.run(tf.global_variables_initializer())

        def extract(p):
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

            return IOS

        def evaluate(P, Y):
            '''
                计算变长的 准确率 指标
            :return:
            '''
            p_s, p_o = P  # b,n    b,n
            y_s, y_o = Y
            A = 0
            B = 0
            C = 0
            for p_s_, y_s_, p_o_, y_o_ in zip(p_s, y_s, p_o, y_o):
                # 当前句子的长度
                ps_ = extract(p_s_)
                ts_ = extract(y_s_)
                po_ = extract(p_o_)
                to_ = extract(y_o_)
                # 主题错了，全部算错，对了才计算
                if len(ps_) and len(ts_) and ps_[0] == ts_[0]:
                    comm = [i for i in po_ if i in to_]
                    A += len(comm)
                else:
                    pass
                B += len(po_)
                C += len(to_)
                # 计算 acc
            return A, B, C

        def train_step(x, pos, s1, s2, subject_tag, object_tag):
            feed_dict = {
                ner.input_chars: x,
                ner.input_pos: pos,
                ner.s1: s1,
                ner.s2: s2,
                ner.subject_tag: subject_tag,
                ner.object_tag: object_tag,
                ner.is_training: True,
            }
            _, step, p_s, p_o, cost_s, cost_o = sess.run(
                [train_op, global_step, ner.p_s, ner.p_o, ner.loss_s, ner.loss_o],
                feed_dict)
            tp, p_, r_ = evaluate((p_s, p_o,), (subject_tag, object_tag))
            time_str = str(int(time.time()))
            p = float(tp) / p_ if p_ else 0.0
            r = float(tp) / r_ if r_ else 0.0
            f = 2 * p * r / (p + r) if p + r else 0.0
            print("{}: step {}, loss_s {}, loss_o {}, p {}, r {}, f {}".format(time_str, step, cost_s, cost_o, p, r, f))
            # train_summary_writer.add_summary(summaries, step)
            return step

        def dev_step(x, pos, s1, s2, subject_tag, object_tag):
            feed_dict = {
                ner.input_chars: x,
                ner.input_pos: pos,
                ner.s1: s1,
                ner.s2: s2,
                ner.subject_tag: subject_tag,
                ner.object_tag: object_tag,
                ner.is_training: False,
            }
            step, p_s, p_o, cost = sess.run(
                [global_step, ner.p_s, ner.p_o, ner.loss],
                feed_dict)

            tp, p_, r_ = evaluate((p_s, p_o,), (subject_tag, object_tag))

            time_str = str(int(time.time()))
            p = float(tp) / p_ if p_ else 0.0
            r = float(tp) / r_ if r_ else 0.0
            f = 2 * p * r / (p + r) if p + r else 0.0
            print("+dev+{}: step {}, loss {}, p {}, r {}, f {}".format(time_str, step, cost, p, r, f))
            # time_str = str(int(time.time()))
            # print("+dev+{}: step {}, loss {}, f_acc {}, t_acc {}".format(time_str, step, cost, accuracy, acc_d))
            return cost, tp, p_, r_

        best_accuracy, best_at_step = 0, 0

        train_example_len = 173109
        dev_example_len = 21639
        num_train_steps = int(train_example_len / FLAGS.batch_size * FLAGS.num_epochs)
        num_dev_steps = int(dev_example_len / FLAGS.batch_size)

        max_f = json.loads(open("f.json").read())["f"]
        min_loss = json.loads(open("f.json").read())["loss"]

        input_ids_train, input_pos_train, s1_train, s2_train, subject_tag_train, object_tag_train = get_input_data(
            "./train_data_6/train_position.tf_record", FLAGS.batch_size)

        input_ids_dev, input_pos_dev, s1_dev, s2_dev, subject_tag_dev, object_tag_dev = get_input_data(
            "./train_data_6/dev_position.tf_record", FLAGS.batch_size)

        with open("loss.txt", "w") as fw:
            for i in range(num_train_steps):
                # batch 数据
                input_ids_train_, input_pos_train_, s1_train_, s2_train_, subject_tag_train_, object_tag_train_ = sess.run(
                    [input_ids_train, input_pos_train, s1_train, s2_train, subject_tag_train, object_tag_train])
                step = train_step(input_ids_train_, input_pos_train_, s1_train_, s2_train_, subject_tag_train_,
                                  object_tag_train_)
                if step % FLAGS.evaluate_every == 0:
                    # dev 数据过大， 也需要进行 分批
                    TP = 0
                    P_ = 0
                    R_ = 0
                    total_loss = 0
                    for j in range(num_dev_steps):
                        # input_ids_dev_, input_pos_dev_, output_types_dev_ = sess.run(
                        #     [input_ids_dev, input_pos_dev, output_types_dev])
                        input_ids_dev_, input_pos_dev_, s1_dev_, s2_dev_, object_start_dev_, object_end_dev_ = sess.run(
                            [input_ids_dev, input_pos_dev, s1_dev, s2_dev, subject_tag_dev, object_tag_dev])
                        loss, tp, p_, r_ = dev_step(input_ids_dev_, input_pos_dev_, s1_dev_, s2_dev_, object_start_dev_,
                                                    object_end_dev_)
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
                    fw.write("loss {}, p {}, r {}, f {}".format(total_loss, p, r, f))
                    if total_loss < min_loss:
                        print("save model:\t%f\t>%f\t%f\t>%f" % (total_loss, p, r, f))
                        min_loss = total_loss
                        saver.save(sess, './ckpt_7/ner.ckpt', global_step=step)
        open("f.json", "w").write(json.dumps({"f": max_f, "loss": total_loss}))
        sess.close()


class Model:
    def __init__(self):
        output_graph_def = tf.GraphDef()

        with open('./ckpt_7/ner.pb', "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        # 读入数据
        self.char2id, self.ner2id, self.pos2id = load_dict(char_dict="train_data_6/char2id.json",
                                                           schema_dict="train_data_6/schema2id.json",
                                                           pos_dict="train_data_6/pos2id.json")
        self.id2type = {value: key for key, value in self.ner2id.items()}
        self.ids = list(self.id2type.keys())
        self.input_ids = self.sess.graph.get_tensor_by_name("placeholder/input_chars:0")
        self.input_pos = self.sess.graph.get_tensor_by_name("placeholder/input_pos:0")
        # is_training = sess.graph.get_tensor_by_name("placeholder/Placeholder:0")  # is_training
        """
        predict_subject/fully_connected_1/BiasAdd
        predict_subject/Softmax
        predict_subject/Reshape_1/shape
        predict_subject/Reshape_1
        predict_subject/subject/dimension
        predict_subject/subject
        predict_subject/Reshape_2/shape
        predict_subject/Reshape_2
        """
        self.p_o = self.sess.graph.get_tensor_by_name("predict_po/ReverseSequence_1:0")
        self.p_s = self.sess.graph.get_tensor_by_name("predict_subject/fully_connected_1/BiasAdd:0")
        print(self.p_s.shape)


    def extract(self, p):
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

        return IOS

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
        predicts_s, predicts_o = self.sess.run([self.p_s, self.p_o], feed_dict)
        predicts_s = np.argmax(np.reshape(predicts_s, [-1, 384, 3]), 2)

        p_s = predicts_s.tolist()[0]
        # print(p_s)
        subject_ = self.extract(p_s)

        print(subject_)
        subject = ""
        if len(subject_):
            s_l = subject_[0][1] - subject_[0][0]
            s_s = subject_[0][0]
            s_e = subject_[0][1]
            # 取合适的 subject
            for s, e in subject_:
                if e - s > 50:
                    continue
                elif e - s > s_l:
                    s_s = s
                    s_e = e
                else:
                    continue
            # 通过 分词结果纠正一下
            count_index = 0
            for word in words:
                if count_index <= s_s < count_index + len(word["word"]):
                    # 优化
                    if s_e <= count_index + len(word["word"]):
                        s_s = count_index
                        s_e = count_index + len(word["word"])
                        # print("do",s_s,s_e)
                        break
                count_index += len(word["word"])
            subject = text[s_s:s_e]

        extract_dict = []
        p_o = predicts_o.tolist()[0]
        object_ = self.extract(p_o)
        # 首先找到 主题，即 为 2 的
        if subject != "" and len(object_):
            for i in object_:
                if p_o[i[0]] > 1:
                    schema = schemas[p_o[i[0]] - 2]
                    o_ = text[i[0]:i[1]]
                    # {"predicate": "连载网站", "object_type": "网站", "subject_type": "网络小说", "object": "晋江文学城", "subject": "猫喵"}
                    schema["subject"] = subject
                    schema["object"] = o_
                    extract_dict.append(schema)
                    if schema["predicate"] == "丈夫":
                        temp_ = {"object_type": "人物", "predicate": "妻子", "subject_type": "人物"}
                        temp_["subject"] = o_
                        temp_["object"] = subject
                        extract_dict.append(temp_)
                    elif schema["predicate"] == "妻子":
                        temp_ = {"object_type": "人物", "predicate": "丈夫", "subject_type": "人物"}
                        temp_["subject"] = o_
                        temp_["object"] = subject
                        extract_dict.append(temp_)

                    print("ok")
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
    # train()
    schemas = load_schemas()
    m = Model()
    m.submit()
    # # 测试发现，subject 会漏掉一些信息，好奇怪, subject 需要加强
    # with open("./data/train_data.json") as f:
    #     i = 0
    #     for l in f:
    #         a = json.loads(l)
    #         text = a['text']
    #         words = a["postag"]
    #         spo_list = m.predict(text, words)
    #         print(text)
    #         print("predict", spo_list)
    #         print("true", a["spo_list"])
    #         print("\n")
    #         i += 1
    #         if i > 30:
    #             break

    # spo_list = predict("《在温柔中疼痛》是由网络作家宋江创作的一部网络文学作品，发表于2008年，正文语种为简体中文，现已完成") # 测试结果 ok 的, -_-
    # print(spo_list)
    """
    1、妻子和丈夫的实体关系 可以反过来
    2、subject识别不准 可以结合分词的加过修正， 寻找最接近的单词进行 矫正
    3、2-task 任务两步走，（1）先subject再object （2）先object，再subject
    """
    # 妻子和丈夫可以反过来
