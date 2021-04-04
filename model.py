from typing import Dict, Any
import tensorflow as tf
from transformers import TFBertModel
from transformers import BertJapaneseTokenizer
from official.nlp.modeling.ops import beam_search
from official.nlp.transformer import transformer
from official.nlp.transformer import model_utils

START_TOKEN = 2
END_TOKEN = 3
BERT_PRETRAINED_MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
VOCAB_SIZE = 32000


def build_tokenizer() -> BertJapaneseTokenizer:
    tokenizer = BertJapaneseTokenizer.from_pretrained(
        BERT_PRETRAINED_MODEL_NAME)
    return tokenizer


def export(model: tf.keras.models.Model, export_path: str, hidden_size: int):
    class SaveModule(tf.Module):
        def __init__(self, model):
          super(SaveModule, self).__init__()
          self.model = model

        @tf.function
        def serve(self, inputs, memories):
          return self.model.call([inputs, memories], training=False)

    save_module = SaveModule(model)
    inputs_shape = (None, None)
    memories_shape = (None, None, hidden_size)
    signatures = dict(
        serving_default=save_module.serve.get_concrete_function(
            tf.TensorSpec(shape=inputs_shape, dtype=tf.int64, name="inputs"),
            tf.TensorSpec(shape=memories_shape, dtype=tf.float32, name="memories")))
    tf.saved_model.save(save_module, export_path, signatures=signatures)


class MemoryTransformer(transformer.Transformer):
    def __init__(self, params: Dict[str, Any]):

        super(MemoryTransformer, self).__init__(params)
        self.encoder = self.build_encoder()
        self.reminder_stack = transformer.DecoderStack(params)

    @staticmethod
    def build_encoder():
        encoder = TFBertModel.from_pretrained(BERT_PRETRAINED_MODEL_NAME)
        encoder.trainable = False
        return encoder

    def call(self, inputs: list, training: bool) -> Any:
        if len(inputs) == 3:
            inputs, memories, targets = inputs[0], inputs[1], inputs[2]
        else:
            # Decoding path.
            inputs, memories, targets = inputs[0], inputs[1], None

        # Variance scaling is used here because it seems to work in many problems.
        # Other reasonable initializers may also work just as well.
        with tf.name_scope("MemoryTransformer"):
            # Calculate attention bias for encoder self-attention and decoder
            # multi-headed attention layers.
            encoder_attention_bias = model_utils.get_padding_bias(inputs)

            # Run the inputs through the encoder layer to map the symbol
            # representations to continuous representations.
            encoder_outputs = self.encode(inputs, encoder_attention_bias,
                                          training)

            reminder_outputs = self.remind(memories, encoder_outputs,
                                           encoder_attention_bias, training)

            reminder_attention_bias = tf.zeros([1, 1, 1, 1],
                                               dtype=self.params["dtype"])

            # Generate output sequence if targets is None, or return logits if target
            # sequence is known.
            if targets is None:
                return self.predict(reminder_outputs, reminder_attention_bias,
                                    training)
            else:
                logits = self.decode(targets, reminder_outputs,
                                     reminder_attention_bias, training)
                return logits, reminder_outputs

    def encode(self, inputs: tf.Tensor, _attention_bias: tf.Tensor,
               _training: bool) -> tf.Tensor:
        return self.encoder(inputs).last_hidden_state

    def remind(self, memories: tf.Tensor, encoder_outputs: tf.Tensor,
               attention_bias: tf.Tensor, training: bool) -> tf.Tensor:
        with tf.name_scope("remind"):
            memories = tf.cast(memories, self.params["dtype"])
            attention_bias = tf.cast(attention_bias, self.params["dtype"])
            if training:
                memories = tf.nn.dropout(
                    memories, rate=self.params["layer_postprocess_dropout"])

            # We don't use self attention bias for reminder
            reminder_self_attention_bias = tf.zeros([1, 1, 1, 1],
                                                    dtype=self.params["dtype"])

            return self.reminder_stack(
                memories, encoder_outputs, reminder_self_attention_bias,
                attention_bias,
                training=training)

    def decode(self, targets: tf.Tensor, reminder_outputs: tf.Tensor,
               attention_bias: tf.Tensor, training: bool) -> tf.Tensor:
        with tf.name_scope("decode"):
            # Prepare inputs to decoder layers by shifting targets, adding positional
            # encoding and applying dropout.
            decoder_inputs = self.embedding_softmax_layer(targets)
            decoder_inputs = tf.cast(decoder_inputs, self.params["dtype"])
            attention_bias = tf.cast(attention_bias, self.params["dtype"])
            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(decoder_inputs)[1]
                pos_encoding = self.position_embedding(decoder_inputs)
                pos_encoding = tf.cast(pos_encoding, self.params["dtype"])
                decoder_inputs += pos_encoding
            if training:
                decoder_inputs = tf.nn.dropout(
                    decoder_inputs,
                    rate=self.params["layer_postprocess_dropout"])

            # Run values
            decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
                length, dtype=self.params["dtype"])
            outputs = self.decoder_stack(
                decoder_inputs,
                reminder_outputs,
                decoder_self_attention_bias,
                attention_bias,
                training=training)
            logits = self.embedding_softmax_layer(outputs, mode="linear")
            logits = tf.cast(logits, tf.float32)
            return logits

    def predict(self, reminder_outputs: tf.Tensor, attention_bias: tf.Tensor,
                training: bool) -> Dict[str, Any]:
        """Return predicted sequence."""
        reminder_outputs = tf.cast(reminder_outputs, self.params["dtype"])
        if self.params["padded_decode"]:
            batch_size = reminder_outputs.shape.as_list()[0]
            input_length = reminder_outputs.shape.as_list()[1]
        else:
            batch_size = tf.shape(reminder_outputs)[0]
            input_length = tf.shape(reminder_outputs)[1]
        max_decode_length = input_length + self.params["extra_decode_length"]
        attention_bias = tf.cast(attention_bias,
                                 self.params["dtype"])

        symbols_to_logits_fn = self._get_symbols_to_logits_fn(
            max_decode_length, training)

        # Create initial set of IDs that will be passed into symbols_to_logits_fn.
        initial_ids = tf.fill([batch_size], START_TOKEN)

        # Create cache storing decoder attention values for each layer.
        # pylint: disable=g-complex-comprehension
        init_decode_length = (
            max_decode_length if self.params["padded_decode"] else 0)
        num_heads = self.params["num_heads"]
        dim_per_head = self.params["hidden_size"] // num_heads
        cache = {
            "layer_%d" % layer: {
                "k":
                    tf.zeros(
                        [batch_size, init_decode_length, num_heads,
                         dim_per_head],
                        dtype=self.params["dtype"]),
                "v":
                    tf.zeros(
                        [batch_size, init_decode_length, num_heads,
                         dim_per_head],
                        dtype=self.params["dtype"])
            } for layer in range(self.params["num_hidden_layers"])
        }
        # pylint: enable=g-complex-comprehension

        # Add encoder output and attention bias to the cache.
        cache["encoder_outputs"] = reminder_outputs
        cache["encoder_decoder_attention_bias"] = attention_bias

        # Use beam search to find the top beam_size sequences and scores.
        decoded_ids, scores = beam_search.sequence_beam_search(
            symbols_to_logits_fn=symbols_to_logits_fn,
            initial_ids=initial_ids,
            initial_cache=cache,
            vocab_size=self.params["vocab_size"],
            beam_size=self.params["beam_size"],
            alpha=self.params["alpha"],
            max_decode_length=max_decode_length,
            eos_id=END_TOKEN,
            padded_decode=self.params["padded_decode"],
            dtype=self.params["dtype"])

        # Get the top sequence for each batch element
        top_decoded_ids = decoded_ids[:, 0, 1:]
        top_scores = scores[:, 0]

        return {"outputs": top_decoded_ids, "scores": top_scores,
                "memories": reminder_outputs}
