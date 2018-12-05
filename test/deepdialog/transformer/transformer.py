import unittest
import tensorflow as tf
import numpy as np
from deepdialog.transformer.transformer import (
    Transformer, Encoder, Decoder, PAD_ID
)

tf.enable_eager_execution()


class TestTransformer(unittest.TestCase):
    def test_build_graph(self):
        vocab_size = 17
        max_length = 13
        hidden_dim = 32
        with tf.Graph().as_default(), tf.Session() as sess:
            model = Transformer(vocab_size, hopping_num=3, head_num=4, hidden_dim=hidden_dim,
                                dropout_rate=0.1, max_length=max_length)
            model.build_graph()
            sess.run(tf.global_variables_initializer())
            loss, acc, prediction = sess.run([model.loss, model.acc, model.prediction], feed_dict={
                model.is_training: True,
                model.encoder_input: np.array([[10, 11, 12], [13, 14, 15]]),
                model.decoder_input: np.array([[1, 20, 21, 2], [1, 22, 23, 2]]),
            })
            self.assertIsInstance(loss, np.float32)
            self.assertIsInstance(acc, np.float32)
            self.assertEqual(prediction.shape, (2, 3, vocab_size))  # 3 == decoder_len - 1

            # Graph がグローバルに持っている重みと、モデルがプロパティとして持っている重みが一致することのテスト
            graph_weight_set = set(tf.trainable_variables())
            model_weight_set = set(model.weights)
            self.assertEqual(model_weight_set, graph_weight_set)

    def test_call(self):
        vocab_size = 17
        batch_size = 7
        max_length = 13
        enc_length = 11
        dec_length = 10
        hidden_dim = 32
        model = Transformer(vocab_size, hopping_num=3, head_num=4, hidden_dim=hidden_dim,
                            dropout_rate=0.1, max_length=max_length)
        encoder_input = tf.ones(shape=[batch_size, enc_length], dtype=tf.int32)
        decoder_input = tf.ones(shape=[batch_size, dec_length], dtype=tf.int32)
        y = model(
            encoder_input,
            decoder_input,
            training=True,
        )
        self.assertEqual(y.shape, [batch_size, dec_length, vocab_size])

    def test_create_enc_attention_mask(self):
        P = PAD_ID
        x = tf.constant([
            [1, 2, 3, P],
            [1, 2, P, P],
        ])
        model = Transformer(vocab_size=17)
        self.assertEqual(model._create_enc_attention_mask(x).numpy().tolist(), [
            [[[False, False, False, True]]],
            [[[False, False, True, True]]],
        ])

    def test_create_dec_self_attention_mask(self):
        P = PAD_ID
        x = tf.constant([
            [1, 2, 3, P],
            [1, 2, P, P],
        ])
        model = Transformer(vocab_size=17)
        self.assertEqual(model._create_dec_self_attention_mask(x).numpy().tolist(), [
            [[
                [False, True,  True,  True],
                [False, False, True,  True],
                [False, False, False, True],
                [False, False, False, True],
            ]],
            [[
                [False, True,  True, True],
                [False, False, True, True],
                [False, False, True, True],
                [False, False, True, True],
            ]],
        ])


class TestEncoder(unittest.TestCase):
    def test_call(self):
        vocab_size = 17
        batch_size = 7
        max_length = 13
        length = 11
        hidden_dim = 32
        model = Encoder(vocab_size, hopping_num=3, head_num=4, hidden_dim=hidden_dim,
                        dropout_rate=0.1, max_length=max_length)
        x = tf.ones(shape=[batch_size, length], dtype=tf.int32)
        mask = tf.cast(tf.zeros(shape=[batch_size, 1, 1, length]), tf.bool)
        y = model(x, self_attention_mask=mask, training=True)
        self.assertEqual(y.shape, [batch_size, length, hidden_dim])


class TestDecoder(unittest.TestCase):
    def test_call(self):
        vocab_size = 17
        batch_size = 7
        max_length = 13
        enc_length = 11
        dec_length = 10
        hidden_dim = 32
        model = Decoder(vocab_size, hopping_num=3, head_num=4, hidden_dim=hidden_dim,
                        dropout_rate=0.1, max_length=max_length)
        decoder_input = tf.ones(shape=[batch_size, dec_length], dtype=tf.int32)
        encoder_output = tf.ones(shape=[batch_size, enc_length, hidden_dim])
        dec_self_attention_mask = tf.cast(tf.zeros(shape=[batch_size, 1, dec_length, dec_length]), tf.bool)
        enc_dec_attention_mask = tf.cast(tf.zeros(shape=[batch_size, 1, 1, enc_length]), tf.bool)
        y = model(
            decoder_input,
            encoder_output,
            self_attention_mask=dec_self_attention_mask,
            enc_dec_attention_mask=enc_dec_attention_mask,
            training=True,
        )
        self.assertEqual(y.shape, [batch_size, dec_length, vocab_size])
