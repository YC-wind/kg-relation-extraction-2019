#!/usr/bin/python
# coding:utf8
"""
@author: Cong Yu
@time: 2019-04-02 11:40
"""
import os, time
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
from data import get_input_data, load_dict, sequence_padding

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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

    def __init__(self, vocab_size, num_classes, embedding_size=256, hidden_size=32, max_num=256, dropout=0.5):
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
            self.input_ids = tf.placeholder(tf.int32, [None, max_num], name='input_ids')
            self.output_types = tf.placeholder(tf.int32, [None, max_num], name='output_types')
            self.is_training = tf.placeholder(tf.bool)  # , name='is_training'
        self.input_m = tf.count_nonzero(self.input_ids, -1)  # 一维数据
        # 区分 训练和测试
        if self.is_training is True:
            self.dropout = 0.5
        else:
            self.dropout = 1.0
        # 构建模型
        # embedding 层 batch * max_sentence_num (?, 256, 64)
        word_embedded = self.word2vec()
        print("word_embedded", word_embedded.shape)

        # lstm 层 + dropout (?, 256, 128)
        vec_lstm = self.sent2vec_lstm(word_embedded)
        print("vec_lstm", vec_lstm.shape)

        # cnn 层 + dropout (?, 256, 128)
        vec_cnn = self.sent2vec_cnn(word_embedded)
        print("vec_cnn", vec_cnn.shape)

        # concat (?, 256, 512)
        concat_vec = tf.concat((vec_lstm, vec_cnn), 2)
        print("concat_vec", concat_vec.shape)

        # crf 输出层
        self.classifer(concat_vec)
        print("viterbi_sequence", self.viterbi_sequence.shape)

        # 定义损失
        self.loss = -tf.reduce_mean(self.log_likelihood)
        # 计算准确率,这种方法没有考虑 序列的真实长度。
        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.viterbi_sequence, self.output_types), tf.float32))

    def word2vec(self):
        """
            # 嵌入层
        :return:
        """
        with tf.name_scope("embedding"):
            # 从截断的正态分布输出随机值
            embedding_mat = tf.Variable(tf.truncated_normal((self.vocab_size, self.embedding_size)))
            # shape为[batch_size, max_num, embedding_size]
            word_embedded = tf.nn.embedding_lookup(embedding_mat, self.input_ids)
        return word_embedded

    def sent2vec_lstm(self, word_embedded):
        """
            # GRU的输入tensor是[batch_size, max_time, ...].在构造句子向量时max_time应该是每个句子的长度，所以这里将
            *** # batch_size * sent_in_doc当做是batch_size.这样一来，每个GRU的cell处理的都是一个单词的词向量
            # 并最终将一句话中的所有单词的词向量融合（Attention）在一起形成句子向量
        :param word_embedded:
        :return:
        """
        with tf.name_scope("bi-lstm"):
            # shape为[batch_size, max_num, hidden_size*2]
            sent_encoded = self.BidirectionalLSTMEncoder(word_embedded, name='word_encoder')
            # # 添加 dropout 0.8
            # sent_vec = tf.nn.dropout(
            #     sent_encoded,
            #     0.8,
            #     noise_shape=None,
            #     seed=None,
            #     name=None
            # )
            return sent_encoded

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
                out, self.output_types, self.input_m)
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
    char2id, type2id = load_dict(char_dict="train_data/char2id.json", type_dict="train_data/type2id.json")
    # tf.flags.DEFINE_string("data_dir", "data/data.dat", "data directory")
    tf.flags.DEFINE_integer("vocab_size", len(char2id), "vocabulary size")
    tf.flags.DEFINE_integer("num_classes", len(type2id), "number of classes")
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
        ner = NER(vocab_size=FLAGS.vocab_size,
                  num_classes=FLAGS.num_classes,
                  embedding_size=FLAGS.embedding_size,
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
        if not os.path.exists('./ckpt/'):
            os.makedirs("./ckpt/")

        # 恢复模型 / 重新初始化参数
        # model_file = tf.train.latest_checkpoint('./ckpt/')
        ckpt = tf.train.get_checkpoint_state('./ckpt/')
        if ckpt:
            print("load saved model:\t", ckpt.model_checkpoint_path)
            saver = tf.train.Saver()
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("init model...")
            sess.run(tf.global_variables_initializer())

        def evaluate(viterbi_sequence, Y):
            '''
                计算变长的 准确率 指标
            :return:
            '''
            t_all = 0
            t_true = 0
            for p, y in zip(viterbi_sequence, Y):
                # 当前句子的长度
                l = len(np.nonzero(y))
                # 通过两个序列，计算准确率
                t_all += l
                t_true += np.sum(np.equal(p[:l], y[:l]))
            return float(t_true) / float(t_all), t_true, t_all

        def train_step(x, y):
            feed_dict = {
                ner.input_ids: x,
                ner.output_types: y,
                ner.is_training: True,
            }
            _, step, predicts_t, cost, accuracy = sess.run(
                [train_op, global_step, ner.viterbi_sequence, ner.loss, ner.acc],
                feed_dict)
            acc_t, count, total = evaluate(np.array(predicts_t), y)
            time_str = str(int(time.time()))
            print("{}: step {}, loss {}, f_acc {}, t_acc {}".format(time_str, step, cost, accuracy, acc_t))
            # train_summary_writer.add_summary(summaries, step)
            return step

        def dev_step(x, y, writer=None):
            feed_dict = {
                ner.input_ids: x,
                ner.output_types: y,
                ner.is_training: False,
            }
            step, predicts_d, cost, accuracy = sess.run(
                [global_step, ner.viterbi_sequence, ner.loss, ner.acc],
                feed_dict)

            acc_d, count, total = evaluate(np.array(predicts_d), y)
            time_str = str(int(time.time()))
            print("+dev+{}: step {}, loss {}, f_acc {}, t_acc {}".format(time_str, step, cost, accuracy, acc_d))
            return cost, accuracy, count, total

        best_accuracy, best_at_step = 0, 0
        train_example_len = 173109
        dev_example_len = 21639
        num_train_steps = int(train_example_len / FLAGS.batch_size * FLAGS.num_epochs)
        num_dev_steps = int(dev_example_len / FLAGS.batch_size)
        max_acc = 0.0
        input_ids_train, output_types_train = get_input_data("./train_data/train_ner.tf_record", FLAGS.batch_size)
        input_ids_dev, output_types_dev = get_input_data("./train_data/dev_ner.tf_record", FLAGS.batch_size)
        for i in range(num_train_steps):
            # batch 数据
            input_ids_train_, output_types_train_ = sess.run([input_ids_train, output_types_train])
            step = train_step(input_ids_train_, output_types_train_)
            if step % FLAGS.evaluate_every == 0:
                # dev 数据过大， 也需要进行 分批
                total_dev_correct = 0
                total_devs = 0
                for j in range(num_dev_steps):
                    input_ids_dev_, output_types_dev_ = sess.run([input_ids_dev, output_types_dev])
                    loss, acc, count, total = dev_step(input_ids_dev_, output_types_dev_)
                    total_dev_correct += count
                    total_devs += total
                dev_accuracy = float(total_dev_correct) / total_devs
                print("预测：", total_dev_correct)
                print("长度为：", total_devs)
                print("最后预测结果：", dev_accuracy)
                if dev_accuracy > max_acc:
                    print("save model:\t%f\t>%f" % (dev_accuracy, max_acc))
                    max_acc = dev_accuracy
                    saver.save(sess, './ckpt/ner.ckpt', global_step=step)

        sess.close()


def predict(text):
    """

    :return:
    """

    def load_model():
        output_graph_def = tf.GraphDef()

        with open('./ckpt/ner-1.pb', "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        return sess

    # 读入数据
    char2id, type2id = load_dict(char_dict="train_data/char2id.json", type_dict="train_data/type2id.json")
    id2type = {value: key for key, value in type2id.items()}
    ids = list(id2type.keys())
    sess = load_model()

    input_ids = sess.graph.get_tensor_by_name("placeholder/input_ids:0")
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
    print(p)
    print(IOS)
    extract_dict = []
    for i in IOS:
        extract_id = p[i[0]]
        tag = id2type.get(extract_id)
        value = text[i[0]:i[1]]
        extract_dict.append({"type":tag,"value":value})
    return extract_dict


if __name__ == "__main__":
    1
    # with tf.Session() as sess:
    #     ner = NER(vocab_size=7025, num_classes=33, embedding_size=256, hidden_size=128, max_num=384)
    #     tvars = tf.trainable_variables()
    #     opts = sess.graph.get_operations()
    #     for v in opts:
    #         print(v.name)
    #     for v in tvars:
    #         print(v.name)
    # train()
    # 基于看出，可以学出来 ，没有 出版时间 关系时， 时间 实体 不会抽取出来，还是蛮智能的
    extract_dict = predict("《猫喵》是晋江文学城连载的小说，作者是黑蓝色")
    print(extract_dict)
