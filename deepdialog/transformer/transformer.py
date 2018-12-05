import tensorflow as tf
from typing import List
from .common_layer import FeedForwardNetwork, ResidualNormalizationWrapper, LayerNormalization
from .embedding import TokenEmbedding, AddPositionalEncoding
from .attention import MultiheadAttention, SelfAttention
from .metrics import padded_cross_entropy_loss, padded_accuracy

PAD_ID = 0


class Transformer(tf.keras.models.Model):
    '''
    Transformer モデルです。
    '''
    def __init__(
            self,
            vocab_size: int,
            hopping_num: int = 4,
            head_num: int = 8,
            hidden_dim: int = 512,
            dropout_rate: float = 0.1,
            max_length: int = 50,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.hopping_num = hopping_num
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.max_length = max_length

        self.encoder = Encoder(
            vocab_size=vocab_size,
            hopping_num=hopping_num,
            head_num=head_num,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            max_length=max_length,
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            hopping_num=hopping_num,
            head_num=head_num,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            max_length=max_length,
        )

    def build_graph(self, name='transformer') -> None:
        '''
        学習/推論のためのグラフを構築します。
        '''
        with tf.name_scope(name):
            self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')
            # [batch_size, max_length]
            self.encoder_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='encoder_input')
            # [batch_size]
            self.decoder_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='decoder_input')

            logit = self.call(
                encoder_input=self.encoder_input,
                decoder_input=self.decoder_input[:, :-1],  # 入力は EOS を含めない
                training=self.is_training,
            )
            decoder_target = self.decoder_input[:, 1:]  # 出力は BOS を含めない

            self.prediction = tf.nn.softmax(logit, name='prediction')

            with tf.name_scope('metrics'):
                xentropy, weights = padded_cross_entropy_loss(
                    logit, decoder_target, smoothing=0.05, vocab_size=self.vocab_size)
                self.loss = tf.identity(tf.reduce_sum(xentropy) / tf.reduce_sum(weights), name='loss')

                accuracies, weights = padded_accuracy(logit, decoder_target)
                self.acc = tf.identity(tf.reduce_sum(accuracies) / tf.reduce_sum(weights), name='acc')

    def call(self, encoder_input: tf.Tensor, decoder_input: tf.Tensor, training: bool) -> tf.Tensor:
        enc_attention_mask = self._create_enc_attention_mask(encoder_input)
        dec_self_attention_mask = self._create_dec_self_attention_mask(decoder_input)

        encoder_output = self.encoder(
            encoder_input,
            self_attention_mask=enc_attention_mask,
            training=training,
        )
        decoder_output = self.decoder(
            decoder_input,
            encoder_output,
            self_attention_mask=dec_self_attention_mask,
            enc_dec_attention_mask=enc_attention_mask,
            training=training,
        )
        return decoder_output

    def _create_enc_attention_mask(self, encoder_input: tf.Tensor):
        with tf.name_scope('enc_attention_mask'):
            batch_size, length = tf.unstack(tf.shape(encoder_input))
            pad_array = tf.equal(encoder_input, PAD_ID)  # [batch_size, m_length]
            # shape broadcasting で [batch_size, head_num, (m|q)_length, m_length] になる
            return tf.reshape(pad_array, [batch_size, 1, 1, length])

    def _create_dec_self_attention_mask(self, decoder_input: tf.Tensor):
        with tf.name_scope('dec_self_attention_mask'):
            batch_size, length = tf.unstack(tf.shape(decoder_input))
            pad_array = tf.equal(decoder_input, PAD_ID)  # [batch_size, m_length]
            pad_array = tf.reshape(pad_array, [batch_size, 1, 1, length])

            autoregression_array = tf.logical_not(
                tf.matrix_band_part(tf.ones([length, length], dtype=tf.bool), -1, 0))  # 下三角が False
            autoregression_array = tf.reshape(autoregression_array, [1, 1, length, length])

            return tf.logical_or(pad_array, autoregression_array)


class Encoder(tf.keras.models.Model):
    '''
    トークン列をベクトル列にエンコードする Encoder です。
    '''
    def __init__(
            self,
            vocab_size: int,
            hopping_num: int,
            head_num: int,
            hidden_dim: int,
            dropout_rate: float,
            max_length: int,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hopping_num = hopping_num
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.token_embedding = TokenEmbedding(vocab_size, hidden_dim)
        self.add_position_embedding = AddPositionalEncoding()
        self.input_dropout_layer = tf.keras.layers.Dropout(dropout_rate)

        self.attention_block_list: List[List[tf.keras.models.Model]] = []
        for _ in range(hopping_num):
            attention_layer = SelfAttention(hidden_dim, head_num, dropout_rate, name='self_attention')
            ffn_layer = FeedForwardNetwork(hidden_dim, dropout_rate, name='ffn')
            self.attention_block_list.append([
                ResidualNormalizationWrapper(attention_layer, dropout_rate, name='self_attention_wrapper'),
                ResidualNormalizationWrapper(ffn_layer, dropout_rate, name='ffn_wrapper'),
            ])
        self.output_normalization = LayerNormalization()

    def call(
            self,
            input: tf.Tensor,
            self_attention_mask: tf.Tensor,
            training: bool,
    ) -> tf.Tensor:
        '''
        モデルを実行します

        :param input: shape = [batch_size, length]
        :param training: 学習時は True
        :return: shape = [batch_size, length, hidden_dim]
        '''
        # [batch_size, length, hidden_dim]
        embedded_input = self.token_embedding(input)
        embedded_input = self.add_position_embedding(embedded_input)
        query = self.input_dropout_layer(embedded_input, training=training)

        for i, layers in enumerate(self.attention_block_list):
            attention_layer, ffn_layer = tuple(layers)
            with tf.name_scope(f'hopping_{i}'):
                query = attention_layer(query, attention_mask=self_attention_mask, training=training)
                query = ffn_layer(query, training=training)
        # [batch_size, length, hidden_dim]
        return self.output_normalization(query)


class Decoder(tf.keras.models.Model):
    '''
    エンコードされたベクトル列からトークン列を生成する Decoder です。
    '''
    def __init__(
            self,
            vocab_size: int,
            hopping_num: int,
            head_num: int,
            hidden_dim: int,
            dropout_rate: float,
            max_length: int,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hopping_num = hopping_num
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.token_embedding = TokenEmbedding(vocab_size, hidden_dim)
        self.add_position_embedding = AddPositionalEncoding()
        self.input_dropout_layer = tf.keras.layers.Dropout(dropout_rate)

        self.attention_block_list: List[List[tf.keras.models.Model]] = []
        for _ in range(hopping_num):
            self_attention_layer = SelfAttention(hidden_dim, head_num, dropout_rate, name='self_attention')
            enc_dec_attention_layer = MultiheadAttention(hidden_dim, head_num, dropout_rate, name='enc_dec_attention')
            ffn_layer = FeedForwardNetwork(hidden_dim, dropout_rate, name='ffn')
            self.attention_block_list.append([
                ResidualNormalizationWrapper(self_attention_layer, dropout_rate, name='self_attention_wrapper'),
                ResidualNormalizationWrapper(enc_dec_attention_layer, dropout_rate, name='enc_dec_attention_wrapper'),
                ResidualNormalizationWrapper(ffn_layer, dropout_rate, name='ffn_wrapper'),
            ])
        self.output_normalization = LayerNormalization()
        # 注：本家ではここは TokenEmbedding の重みを転地したものを使っている
        self.output_dense_layer = tf.keras.layers.Dense(vocab_size, use_bias=False)

    def call(
            self,
            input: tf.Tensor,
            encoder_output: tf.Tensor,
            self_attention_mask: tf.Tensor,
            enc_dec_attention_mask: tf.Tensor,
            training: bool,
    ) -> tf.Tensor:
        '''
        モデルを実行します

        :param input: shape = [batch_size, length]
        :param training: 学習時は True
        :return: shape = [batch_size, length, hidden_dim]
        '''
        # [batch_size, length, hidden_dim]
        embedded_input = self.token_embedding(input)
        embedded_input = self.add_position_embedding(embedded_input)
        query = self.input_dropout_layer(embedded_input, training=training)

        for i, layers in enumerate(self.attention_block_list):
            self_attention_layer, enc_dec_attention_layer, ffn_layer = tuple(layers)
            with tf.name_scope(f'hopping_{i}'):
                query = self_attention_layer(query, attention_mask=self_attention_mask, training=training)
                query = enc_dec_attention_layer(query, memory=encoder_output,
                                                attention_mask=enc_dec_attention_mask, training=training)
                query = ffn_layer(query, training=training)

        query = self.output_normalization(query)  # [batch_size, length, hidden_dim]
        return self.output_dense_layer(query)  # [batch_size, length, vocab_size]
