import os
import glob
import tensorflow as tf
import pandas as pd
import numpy as np

import model

BUFFER_SIZE = 20000


class DataLoader:
    def __init__(self, len_conversations: int, max_token_length: int,
                 dataset_dir: str, batch_size: int):
        self.len_conversations = len_conversations
        self.max_token_length = max_token_length
        self.tokenizer = model.build_tokenizer()

        df = self.read_dir(dataset_dir)
        train_data = self.parse_df(df)
        self.steps_per_epoch = self.get_steps_per_epoch(train_data, batch_size)
        self.dataset = self.build_dataset(train_data, batch_size, is_train=True)

    @staticmethod
    def read_dir(dataset_dir: str) -> pd.DataFrame:
        file_pattern = os.path.join(dataset_dir, '**/*.tsv')
        file_paths = glob.glob(file_pattern, recursive=True)
        df = pd.DataFrame()
        for path in file_paths:
            file_df = pd.read_table(path, names=(
                'conversation_id', 'status_id', 'original_id', 'text'))
            df = pd.concat([df, file_df], axis=0)

        return df

    def parse_df(self, df: pd.DataFrame) -> np.ndarray:
        num_conversations = len(df.groupby('conversation_id'))
        text_data = df['text'].values
        text_data = text_data.reshape(num_conversations, -1)
        text_data = text_data[:, :self.len_conversations]
        return text_data

    def encode(self, langs: tf.Tensor) -> tf.Tensor:
        tokens = [self.tokenizer.encode(lang.numpy().decode('utf-8'))
                  for lang in langs]
        return tokens

    def tf_encode(self, input: tf.Tensor):
        tout = [tf.int64 for _ in range(self.len_conversations)]
        result = tf.py_function(self.encode, [input], tout)
        for i in range(self.len_conversations):
            result[i].set_shape([None])

        return result

    def filter_max_length(self, *inp) -> bool:
        is_passed = True
        for i in range(self.len_conversations):
            is_passed = tf.logical_and(
                is_passed, tf.size(inp[i]) <= self.max_token_length)
        return is_passed

    def get_steps_per_epoch(self, data: np.ndarray, batch_size: int) -> int:
        one_epoch_ds = self.build_dataset(data, batch_size, is_train=False)
        steps_per_epoch = len([1 for _ in one_epoch_ds])
        return steps_per_epoch

    def build_dataset(self, data: np.ndarray, batch_size: int,
                      is_train: bool) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.map(
            self.tf_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.filter(self.filter_max_length)

        if is_train:
            dataset = dataset.cache()
            dataset = dataset.shuffle(BUFFER_SIZE)
            num_repeat = None
        else:
            num_repeat = 1

        dataset = dataset.padded_batch(batch_size)
        dataset = dataset.repeat(num_repeat)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

