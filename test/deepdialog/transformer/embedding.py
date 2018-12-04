import unittest
import tensorflow as tf
import numpy as np
import itertools
from tfsample.transformer.embedding import TokenEmbedding, AddPositionalEncoding

tf.enable_eager_execution()


class TestTokenEmbedding(unittest.TestCase):
    def test_call(self):
        vocab_size = 3
        embedding_dim = 4
        layer = TokenEmbedding(vocab_size=vocab_size, embedding_dim=embedding_dim)
        embedded = layer(tf.constant([[0, 1, 2]]))
        embedded_tokens = embedded[0]
        self.assertEqual(embedded_tokens[0].numpy().tolist(), [0] * embedding_dim)
        self.assertNotEqual(embedded_tokens[1].numpy().tolist(), [0] * embedding_dim)


class TestAddPositionalEncoding(unittest.TestCase):
    def test_call(self):
        max_length = 2
        batch_size = 3
        depth = 7

        layer = AddPositionalEncoding()
        input = tf.ones(shape=[batch_size, max_length, depth])
        result = layer(input)
        self.assertEqual(result.shape, [batch_size, max_length, depth])
        positional_encoding = (result - input).numpy()

        # PE_{pos, 2i}   = sin(pos / 10000^{2i / d_model})
        # PE_{pos, 2i+1} = cos(pos / 10000^{2i / d_model})
        for batch, i, pos in itertools.product(range(batch_size), range(depth // 2), range(max_length)):
            self.assertAlmostEqual(
                positional_encoding[batch, pos, i * 2],
                np.sin(pos / 10000 ** (i * 2 / depth)),
                places=6,
            )
            self.assertAlmostEqual(
                positional_encoding[batch, pos, i * 2 + 1],
                np.cos(pos / 10000 ** (i * 2 / depth)),
                places=6,
            )

    def test_call_graph(self):
        batch_size = 3
        max_length = 5
        depth = 7
        data = np.ones(shape=[batch_size, max_length, depth])

        with tf.Graph().as_default():
            with tf.Session() as sess:
                layer = AddPositionalEncoding()
                input = tf.placeholder(shape=[None, None, None], dtype=tf.float32)
                result_op = layer(input)
                result = sess.run(result_op, feed_dict={
                    input: data,
                })
                self.assertEqual(result.shape, (batch_size, max_length, depth))

        positional_encoding = result - data

        # PE_{pos, 2i}   = sin(pos / 10000^{2i / d_model})
        # PE_{pos, 2i+1} = cos(pos / 10000^{2i / d_model})
        for batch, i, pos in itertools.product(range(batch_size), range(depth // 2), range(max_length)):
            self.assertAlmostEqual(
                positional_encoding[batch, pos, i * 2],
                np.sin(pos / 10000 ** (i * 2 / depth)),
                places=6,
            )
            self.assertAlmostEqual(
                positional_encoding[batch, pos, i * 2 + 1],
                np.cos(pos / 10000 ** (i * 2 / depth)),
                places=6,
            )
