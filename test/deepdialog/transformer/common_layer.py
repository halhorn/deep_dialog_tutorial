import unittest
import tensorflow as tf
import numpy as np
from deepdialog.transformer.common_layer import (
    FeedForwardNetwork, LayerNormalization, ResidualNormalizationWrapper
)

tf.enable_eager_execution()


class TestFeedForwardNetwork(unittest.TestCase):
    def test_call(self):
        batch_size = 3
        length = 5
        hidden_dim = 32
        input_dim = 16
        model = FeedForwardNetwork(hidden_dim, dropout_rate=0.1)
        x = tf.ones(shape=[batch_size, length, input_dim])
        result = model(x, training=True)
        self.assertEqual(result.shape, [batch_size, length, hidden_dim])


class TestResidualNormalization(unittest.TestCase):
    def test_call(self):
        batch_size = 3
        length = 5
        hidden_dim = 32
        layer = FeedForwardNetwork(hidden_dim, dropout_rate=0.1)
        wrapped_layer = ResidualNormalizationWrapper(layer, dropout_rate=0.1)

        x = tf.ones(shape=[batch_size, length, hidden_dim])
        y = wrapped_layer(x, training=True)
        self.assertEqual(y.shape, [batch_size, length, hidden_dim])


class TestLayerNormalization(unittest.TestCase):
    def test_call(self):
        x = tf.constant([[0, 4], [-2, 2]], dtype=tf.float32)
        layer = LayerNormalization()
        y = layer(x)
        expect = [[-1, 1], [-1, 1]]
        for y1, e1 in zip(y.numpy(), expect):
            for y2, e2 in zip(y1, e1):
                self.assertAlmostEqual(y2, e2, places=5)

    def test_call_graph(self):
        batch_size = 2
        length = 3
        hidden_dim = 5

        with tf.Graph().as_default(), tf.Session() as sess:
            layer = LayerNormalization()
            x = tf.placeholder(dtype=tf.float32, shape=[None, None, hidden_dim])
            y = layer(x)
            sess.run(tf.global_variables_initializer())
            sess.run(y, feed_dict={x: np.ones(shape=[batch_size, length, hidden_dim])})
