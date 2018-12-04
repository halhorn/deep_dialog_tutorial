import unittest
from deepdialog.transformer.preprocess.batch_generator import BatchGenerator, ENCODER_INPUT_NODE, DECODER_INPUT_NODE


class TestBatchGenerator(unittest.TestCase):
    def test_vocab_size(self):
        batch_generator = BatchGenerator()
        self.assertEqual(batch_generator.vocab_size, 8000)

    def test_get_batch(self):
        batch_generator = BatchGenerator()
        batch_generator.data = [
            ([1, 2, 3], [1, 4, 5, 2]),
            ([4, 5], [1, 6, 7, 8, 2]),
        ]
        gen = batch_generator.get_batch(batch_size=2, shuffle=False)
        result = gen.__next__()
        self.assertEqual(result[ENCODER_INPUT_NODE].tolist(), [
            [1, 2, 3],
            [4, 5, 0],
        ])
        self.assertEqual(result[DECODER_INPUT_NODE].tolist(), [
            [1, 4, 5, 2, 0],
            [1, 6, 7, 8, 2],
        ])

    def test_create_data(self):
        batch_generator = BatchGenerator()
        lines = ['こんにちは', 'やあ', 'いい天気だ']
        result = batch_generator._create_data(lines)
        self.assertEqual(len(result), 2)
        self.assertEqual(batch_generator.sp.decode_ids(result[0][0]), 'こんにちは')

    def test_create_question(self):
        batch_generator = BatchGenerator()
        ids = batch_generator._create_question('こんにちは')
        self.assertEqual(batch_generator.sp.decode_ids(ids), 'こんにちは')

    def test_create_answer(self):
        batch_generator = BatchGenerator()
        ids = batch_generator._create_answer('こんにちは')
        self.assertEqual(batch_generator.sp.id_to_piece(ids[0]), '<s>')
        self.assertEqual(batch_generator.sp.id_to_piece(ids[-1]), '</s>')
        self.assertEqual(batch_generator.sp.decode_ids(ids), 'こんにちは')

    def test_split(self):
        batch_generator = BatchGenerator()

        for data in (
                [0, 1, 2, 3, 4, 5, 6, 7, 8],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ):
            splited = batch_generator._split(data, 3)
            self.assertEqual(splited, [
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
            ], 'test with {0}'.format(data))

    def test_convert_to_array(self):
        batch_generator = BatchGenerator()
        id_list_list = [
            [1, 2],
            [3, 4, 5, 6],
            [7],
        ]
        self.assertEqual(batch_generator._convert_to_array(id_list_list).tolist(), [
            [1, 2, 0, 0],
            [3, 4, 5, 6],
            [7, 0, 0, 0],
        ])
