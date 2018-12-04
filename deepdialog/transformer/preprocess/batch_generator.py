import sentencepiece as spm
import numpy as np
import random
from typing import List, Sequence, Tuple

ENCODER_INPUT_NODE = 'transformer/encoder_input:0'
DECODER_INPUT_NODE = 'transformer/decoder_input:0'
IS_TRAINING_NODE = 'transformer/is_training:0'


class BatchGenerator:
    def __init__(
            self,
            max_length=50,
            spm_model_path: str = 'deepdialog/transformer/preprocess/spm_natsume.model'
    ) -> None:
        self.max_length = max_length
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(spm_model_path)
        self.bos = self.sp.piece_to_id('<s>')
        self.eos = self.sp.piece_to_id('</s>')
        self.pad = 0

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def load(self, file_path: str) -> None:
        with open(file_path) as f:
            lines = [line.strip() for line in f.readlines()]
        self.data = self._create_data(lines)

    def get_batch(self, batch_size: int = 128, shuffle=True):
        while True:
            if shuffle:
                random.shuffle(self.data)
            raw_batch_list = self._split(self.data, batch_size)
            for raw_batch in raw_batch_list:
                questions, answers = zip(*raw_batch)
                yield {
                    ENCODER_INPUT_NODE: self._convert_to_array(questions),
                    DECODER_INPUT_NODE: self._convert_to_array(answers),
                    IS_TRAINING_NODE: True,
                }

    def _create_data(self, lines: Sequence[str]) -> List[Tuple[List[int], List[int]]]:
        questions = [self._create_question(line) for line in lines[:-1]]
        answers = [self._create_answer(line) for line in lines[1:]]
        return list(zip(questions, answers))

    def _create_question(self, sentence) -> List[int]:
        ids = self.sp.encode_as_ids(sentence)
        return ids[:self.max_length]

    def _create_answer(self, sentence: str) -> List[int]:
        ids = self.sp.encode_as_ids(sentence)
        return [self.bos] + ids[:self.max_length - 2] + [self.eos]

    def _split(self, nd_list: Sequence, batch_size: int) -> List[List]:
        return [list(nd_list[i - batch_size:i]) for i in range(batch_size, len(nd_list) + 1, batch_size)]

    def _convert_to_array(self, id_list_list: Sequence[Sequence[int]]) -> np.ndarray:
        max_len = max([len(id_list) for id_list in id_list_list])

        return np.array(
            [list(id_list) + [self.pad] * (max_len - len(id_list)) for id_list in id_list_list],
            dtype=np.int32,
        )
