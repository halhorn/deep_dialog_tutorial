import unittest
import tensorflow as tf
from deepdialog.transformer import metrics

tf.enable_eager_execution()


class TestMetrics(unittest.TestCase):
    def test_padded_cross_entropy_loss(self):
        logits = tf.constant([[
            [0.1, -0.1, 1., 0., 0.],
            [0.1, -0.1, 1., 0., 0.],
        ]])
        labels = tf.constant([[2, 2]])
        metrics.padded_cross_entropy_loss(logits, labels, smoothing=0.05, vocab_size=5)

    def test_padded_accuracy(self):
        logits = tf.constant([[
            [0.1, -0.1, 1., 0., 0.],
            [0.1, -0.1, 1., 0., 0.],
            [0.1, -0.1, 1., 0., 0.],
        ]])
        labels = tf.constant([[2, 3, 0]])  # 0 == PAD
        result, weight = metrics.padded_accuracy(logits, labels)
        self.assertEqual(result.numpy().tolist(), [[1., 0., 0.]])
        self.assertEqual(weight.numpy().tolist(), [[1., 1., 0.]])
