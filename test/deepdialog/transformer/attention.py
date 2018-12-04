import unittest
import tensorflow as tf
import numpy as np
from tfsample.transformer.attention import MultiheadAttention, SelfAttention, SimpleAttention

tf.enable_eager_execution()


class TestMultiheadAttention(unittest.TestCase):
    def test_call(self):
        batch_size = 3
        q_length = 5
        m_length = 7
        hidden_dim = 32
        head_num = 4
        with tf.Graph().as_default(), tf.Session() as sess:
            q = tf.placeholder(dtype=tf.float32, shape=[None, None, hidden_dim])
            k = tf.placeholder(dtype=tf.float32, shape=[None, None, hidden_dim])

            mask_numpy = np.zeros(shape=[batch_size, 1, q_length, m_length])
            mask_numpy[0, 0, :, -1] = 1
            mask = tf.constant(mask_numpy, dtype=tf.bool)

            model = MultiheadAttention(hidden_dim=hidden_dim, head_num=head_num, dropout_rate=0.1)
            result_op = model(q, k, mask, training=True)
            sess.run(tf.global_variables_initializer())
            result, attention_weight = sess.run([result_op, 'multihead_attention/attention_weight:0'], feed_dict={
                q: np.ones(shape=[batch_size, q_length, hidden_dim]),
                k: np.ones(shape=[batch_size, m_length, hidden_dim]),
            })
            self.assertEqual(result.shape, (batch_size, q_length, hidden_dim))
            self.assertEqual(attention_weight[0, 0, :, -1].tolist(), [0.0] * q_length)

    def test_split_head(self):
        batch_size = 3
        length = 5
        hidden_dim = 32
        head_num = 4
        model = MultiheadAttention(hidden_dim=hidden_dim, head_num=head_num, dropout_rate=0.1)
        x = tf.ones(shape=[batch_size, length, hidden_dim])
        y = model._split_head(x)
        self.assertEqual(y.shape, [batch_size, head_num, length, hidden_dim // head_num])

    def test_split_head_graph(self):
        batch_size = 3
        length = 5
        hidden_dim = 32
        head_num = 4
        with tf.Graph().as_default(), tf.Session() as sess:
            model = MultiheadAttention(hidden_dim=hidden_dim, head_num=head_num, dropout_rate=0.1)
            x = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, None])
            y = model._split_head(x)
            sess.run(y, feed_dict={x: np.ones(shape=[batch_size, length, hidden_dim])})

    def test_combine_head(self):
        batch_size = 3
        length = 5
        hidden_dim = 32
        head_num = 4
        model = MultiheadAttention(hidden_dim=hidden_dim, head_num=head_num, dropout_rate=0.1)
        x = tf.ones(shape=[batch_size, head_num, length, hidden_dim // head_num])
        y = model._combine_head(x)
        self.assertEqual(y.shape, [batch_size, length, hidden_dim])

        x = tf.reshape(tf.range(batch_size * length * hidden_dim), [batch_size, length, hidden_dim])
        reconstructed = model._combine_head(model._split_head(x))
        self.assertEqual(reconstructed.numpy().tolist(), x.numpy().tolist())


class TestSelfAttention(unittest.TestCase):
    def test_call(self):
        batch_size = 3
        q_length = 5
        hidden_dim = 32
        head_num = 4
        q = tf.ones(shape=[batch_size, q_length, hidden_dim])
        mask = tf.zeros(shape=[batch_size, 1, 1, q_length])
        model = SelfAttention(hidden_dim=hidden_dim, head_num=head_num, dropout_rate=0.1)
        result = model(q, mask, training=True)
        self.assertEqual(result.shape, [batch_size, q_length, hidden_dim])


class TestSimpleAttention(unittest.TestCase):
    def test_call(self):
        batch_size = 3
        q_length = 5
        m_length = 7
        depth = 32

        model = SimpleAttention(depth=depth)
        query = tf.ones(shape=[batch_size, q_length, depth])
        memory = tf.ones(shape=[batch_size, m_length, depth])
        result = model(query, memory)
        self.assertEqual(result.shape, [batch_size, q_length, depth])
