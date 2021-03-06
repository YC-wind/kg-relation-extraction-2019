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
from data_4 import get_input_data, load_dict, sequence_padding

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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


class NER:
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
            self.input_chars = tf.placeholder(tf.int32, [None, max_num], name='input_chars')
            self.input_pos = tf.placeholder(tf.int32, [None, max_num], name='input_pos')
            self.output = tf.placeholder(tf.int32, [None, max_num], name='output')
            self.is_training = tf.placeholder(tf.bool, name='is_training')  # , name='is_training'
        self.input_m = tf.count_nonzero(self.input_chars, -1)  # 一维数据
        # 区分 训练和测试
        if self.is_training is True:
            self.dropout = 0.5
        else:
            self.dropout = 1.0
        # 构建模型
        # embedding 层 batch * max_sentence_num (?, 256, 64)
        word_embedded = self.word2vec()
        print("word_embedded", word_embedded.shape)

        pos_embedded = self.pos2vec()
        print("pos_embedded", pos_embedded.shape)

        # lstm 层 + dropout (?, 256, 128)
        vec_lstm_c = self.BidirectionalLSTMEncoder(word_embedded, name="bi-lstm-c")
        print("vec_lstm_c", vec_lstm_c.shape)

        vec_lstm_p = self.BidirectionalLSTMEncoder(pos_embedded, name="bi-lstm-p")
        print("vec_lstm_p", vec_lstm_p.shape)

        add_vec = tf.add_n([vec_lstm_c, vec_lstm_p]) / 2

        print("add_vec", add_vec.shape)

        # crf 输出层
        self.classifer(add_vec)
        print("viterbi_sequence", self.viterbi_sequence.shape)

        # 定义损失
        self.loss = -tf.reduce_mean(self.log_likelihood)
        # 计算准确率,这种方法没有考虑 序列的真实长度。
        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.viterbi_sequence, self.output), tf.float32))

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

    def sent2vec_cnn(self, word_embedded):
        """
            # GRU的输入tensor是[batch_size, max_time, ...].在构造句子向量时max_time应该是每个句子的长度，所以这里将
            *** # batch_size * sent_in_doc当做是batch_size.这样一来，每个GRU的cell处理的都是一个单词的词向量
            # 并最终将一句话中的所有单词的词向量融合（Attention）在一起形成句子向量
        :param word_embedded:
        :return:
        """
        with tf.name_scope("cnn"):
            cnn_1 = tf.layers.conv1d(word_embedded, self.hidden_size, 2, padding="same", activation=tf.nn.relu)
            # 添加 dropout 0.8
            dropout_1 = tf.nn.dropout(
                cnn_1,
                self.dropout,
                noise_shape=None,
                seed=None,
                name=None
            )
            cnn_2 = tf.layers.conv1d(word_embedded, self.hidden_size, 3, padding="same", activation=tf.nn.relu)
            # 添加 dropout 0.8
            dropout_2 = tf.nn.dropout(
                cnn_2,
                self.dropout,
                noise_shape=None,
                seed=None,
                name=None
            )
            cnn_3 = tf.layers.conv1d(word_embedded, self.hidden_size, 4, padding="same", activation=tf.nn.relu)
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

    def BidirectionalGRUEncoder(self, inputs, name):
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
            GRU_cell_fw = rnn.GRUCell(self.hidden_size)
            GRU_cell_bw = rnn.GRUCell(self.hidden_size)
            # fw_outputs和bw_outputs的size都是[batch_size, max_time, hidden_size]
            ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,
                                                                                 cell_bw=GRU_cell_bw,
                                                                                 inputs=inputs,
                                                                                 sequence_length=length(inputs),
                                                                                 dtype=tf.float32)
            # outputs的size是[batch_size, max_time, hidden_size*2] =
            # [batch_size * sent_in_doce, word_in_sent, hidden_size*2]
            outputs = tf.concat((fw_outputs, bw_outputs), 2)
            return outputs

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
            # u_context是上下文的重要性向量，用于区分不同单词/句子对于句子/文档的重要程度,
            # 因为使用双向GRU，所以其长度为2×hidden_szie( 也是一个变量，之前以为lstm结果 )
            u_context = tf.Variable(tf.truncated_normal([self.hidden_size * 2]), name='u_context')
            # 使用一个全连接层编码GRU的输出的到期 隐层表示,输出u的size是[batch_size, max_time, hidden_size * 2]
            h = layers.fully_connected(inputs, self.hidden_size * 2, activation_fn=tf.nn.tanh)

            # shape为[batch_size, max_time, 1] = [batch_size * sent_in_doc, word_in_sent, 1]
            a = tf.reduce_sum(tf.multiply(h, u_context), axis=2, keepdims=True)
            alpha = tf.nn.softmax(a, axis=1)
            # reduce_sum之前shape为[batch_szie, max_time, hidden_szie*2]，之后shape为[batch_size, hidden_size*2]
            atten_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
            return atten_output


def train():
    """
        模型训练
    :return:
    """
    char2id, ner2id, pos2id = load_dict(char_dict="train_data_4/char2id.json", ner_dict="train_data_4/ner2id.json",
                                        pos_dict="train_data_4/pos2id.json")
    # tf.flags.DEFINE_string("data_dir", "data/data.dat", "data directory")
    tf.flags.DEFINE_integer("vocab_size_c", len(char2id), "vocabulary size")
    tf.flags.DEFINE_integer("vocab_size_p", len(pos2id), "vocabulary size")
    tf.flags.DEFINE_integer("num_classes", len(ner2id), "number of classes")
    tf.flags.DEFINE_integer("max_num", 384, "max_sentence_num")
    tf.flags.DEFINE_integer("embedding_size_c", 256, "Dimensionality of character embedding (default: 200)")
    tf.flags.DEFINE_integer("embedding_size_p", 256, "Dimensionality of character embedding (default: 200)")
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
        ner = NER(vocab_size_c=FLAGS.vocab_size_c,
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
        if not os.path.exists('./ckpt_3/'):
            os.makedirs("./ckpt_3/")

        # 恢复模型 / 重新初始化参数
        # model_file = tf.train.latest_checkpoint('./ckpt/')
        ckpt = tf.train.get_checkpoint_state('./ckpt_3/')
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

        def evaluate(viterbi_sequence, Y):
            '''
                计算变长的 准确率 指标
            :return:
            '''
            TP = 0
            P_ = 0
            R_ = 0
            for p, y in zip(viterbi_sequence, Y):
                # 当前句子的长度
                pre_ = extract(p)
                tru_ = extract(y)
                # 计算 acc
                comm = [i for i in pre_ if i in tru_]
                TP += len(comm)
                P_ += len(pre_)
                R_ += len(tru_)
                # l = len(np.nonzero(y))
                # # 通过两个序列，计算准确率
                # t_all += l
                # t_true += np.sum(np.equal(p[:l], y[:l]))

            return TP, P_, R_

        def train_step(x, pos, y):
            feed_dict = {
                ner.input_chars: x,
                ner.input_pos: pos,
                ner.output: y,
                ner.is_training: True,
            }
            _, step, predicts_t, cost, accuracy = sess.run(
                [train_op, global_step, ner.viterbi_sequence, ner.loss, ner.acc],
                feed_dict)
            tp, p_, r_ = evaluate(np.array(predicts_t), y)
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

        def dev_step(x, pos, y, writer=None):
            feed_dict = {
                ner.input_chars: x,
                ner.input_pos: pos,
                ner.output: y,
                ner.is_training: False,
            }
            step, predicts_d, cost, accuracy = sess.run(
                [global_step, ner.viterbi_sequence, ner.loss, ner.acc],
                feed_dict)

            tp, p_, r_ = evaluate(np.array(predicts_d), y)

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

        input_ids_train, input_pos_train, output_types_train = get_input_data("./train_data_4/train_ner.tf_record",
                                                                              FLAGS.batch_size)
        input_ids_dev, input_pos_dev, output_types_dev = get_input_data("./train_data_4/dev_ner.tf_record",
                                                                        FLAGS.batch_size)
        for i in range(num_train_steps):
            # batch 数据
            input_ids_train_, input_pos_train_, output_types_train_ = sess.run(
                [input_ids_train, input_pos_train, output_types_train])
            step = train_step(input_ids_train_, input_pos_train_, output_types_train_)
            if step % FLAGS.evaluate_every == 0:
                # dev 数据过大， 也需要进行 分批
                TP = 0
                P_ = 0
                R_ = 0
                total_loss = 0
                for j in range(num_dev_steps):
                    input_ids_dev_, input_pos_dev_, output_types_dev_ = sess.run(
                        [input_ids_dev, input_pos_dev, output_types_dev])
                    loss, tp, p_, r_ = dev_step(input_ids_dev_, input_pos_dev_, output_types_dev_)
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
                    saver.save(sess, './ckpt_3/ner.ckpt', global_step=step)

        sess.close()


class Model:
    def __init__(self):
        output_graph_def = tf.GraphDef()

        with open('./ckpt_3/ner.pb', "rb") as f:
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

        with open('./ckpt_3/ner.pb', "rb") as f:
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
    #     ner = NER(vocab_size_c=7025, num_classes=33, embedding_size_c=256, hidden_size=128, max_num=384)
    #     tvars = tf.trainable_variables()
    #     opts = sess.graph.get_operations()
    #     for v in opts:
    #         print(v.name)
    #     for v in tvars:
    #         print(v.name)
    # train()
    schemas = load_schemas()
    m = Model()
    # spo_list = m.predict("作品赏析胡新良作品集胡新良作品中共邵阳县委党校胡新良，男，汉族，1968年7月生，湖南邵阳人",[{"word": "作品", "pos": "n"}, {"word": "赏析", "pos": "v"}, {"word": "胡新良", "pos": "nr"}, {"word": "作品集", "pos": "n"}, {"word": "胡新良", "pos": "nr"}, {"word": "作品", "pos": "n"}, {"word": "中共邵阳县委党校", "pos": "nt"}, {"word": "胡新良", "pos": "nr"}, {"word": "，", "pos": "w"}, {"word": "男", "pos": "a"}, {"word": "，", "pos": "w"}, {"word": "汉族", "pos": "nz"}, {"word": "，", "pos": "w"}, {"word": "1968年", "pos": "t"}, {"word": "7月", "pos": "t"}, {"word": "生", "pos": "v"}, {"word": "，", "pos": "w"}, {"word": "湖南", "pos": "ns"}, {"word": "邵阳", "pos": "ns"}, {"word": "人", "pos": "n"}])
    # print(spo_list)
    m.submit()
    # spo_list = predict("《在温柔中疼痛》是由网络作家宋江创作的一部网络文学作品，发表于2008年，正文语种为简体中文，现已完成") # 测试结果 ok 的, -_-
    # print(spo_list)
