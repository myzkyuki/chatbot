import argparse
import tensorflow as tf

from model import build_tokenizer


class Chat:
    def __init__(self, model_path: str, max_token_length: int,
                 hidden_size: int, disable_memory: bool):
        self.saved_model = tf.saved_model.load(model_path)
        self.tokenizer = build_tokenizer()
        self.memory = tf.zeros((1, max_token_length, hidden_size),
                               dtype=tf.float32)
        self.disable_memory = disable_memory

    def get_reply(self, sentence):
        serve = self.saved_model.signatures["serving_default"]
        inputs = self.tokenizer.encode(sentence)
        inputs = tf.expand_dims(inputs, 0)
        inputs = tf.cast(inputs, dtype=tf.int64)
        result = serve(inputs=inputs, memories=self.memory)
        reply = self.tokenizer.decode(
            [i for i in result['outputs'][0] if i > 3])
        reply = reply.replace(' ', '')
        if not self.disable_memory:
            self.memory = result['memories']

        return reply

    def talk(self):
        while True:
            sentence = input('YOU: ')
            reply = self.get_reply(sentence)
            print('BOT: ' + reply)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--max_token_length', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--disable_memory', action='store_true')
    args = parser.parse_args()

    chat = Chat(model_path=args.model_path,
                max_token_length=args.max_token_length,
                hidden_size=args.hidden_size,
                disable_memory=args.disable_memory)
    chat.talk()


