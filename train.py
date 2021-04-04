import os
import time
import argparse
import functools
import tensorflow as tf
from official.nlp.transformer import metrics
from official.nlp.transformer import optimizer

from logger import logger
from dataloader import DataLoader
from model import MemoryTransformer, export, VOCAB_SIZE


def train(epochs: int, model: tf.keras.models.Model,
          loss_fn: tf.keras.losses.Loss,
          opt: tf.keras.optimizers.Optimizer,
          dataloader: DataLoader, log_dir: str, hidden_size: int):
    total_steps = epochs * dataloader.steps_per_epoch
    start_steps = 0
    checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    ckpt_manager = tf.train.CheckpointManager(
        checkpoint, directory=ckpt_dir, max_to_keep=1)
    if ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint)
        start_steps = opt.iterations.numpy()
        logger.info(f'Loaded checkpoint: {ckpt_manager.latest_checkpoint}')

    logger.info(f'Start steps: {start_steps}')
    logger.info(f'Steps per epoch: {dataloader.steps_per_epoch}')
    logger.info(f'Total steps: {total_steps}')

    train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    tboard_dir = os.path.join(log_dir, 'tensorboard')
    summary_writer = tf.summary.create_file_writer(tboard_dir)

    train_ds_iter = iter(dataloader.dataset)
    train_step_signature = [[
        tf.TensorSpec(shape=(None, None), dtype=tf.int64)
        for _ in range(dataloader.len_conversations - 1)
    ]]

    @tf.function(input_signature=train_step_signature)
    def train_step(inputs):
        batch_size = tf.shape(inputs[0])[0]
        mem = tf.zeros((batch_size, dataloader.max_token_length, hidden_size), dtype=tf.float32)

        with tf.GradientTape() as tape:
            loss = 0
            for i in range(0, len(inputs), 2):
                tar_inp, tar_real = inputs[i + 1][:, :-1], inputs[i + 1][:, 1:]
                logits, mem = model([inputs[i], mem, tar_inp], True)
                loss += loss_fn(logits, tar_real)

        gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss_metric(loss)

        # Calc accuracy with last conversation
        train_accuracy_metric(tar_real, logits)

    logging_step = dataloader.steps_per_epoch // 20
    start = time.time()
    for step in range(start_steps, total_steps):
        inputs = next(train_ds_iter)

        # train reply for 0->1, 2->3, 4->5
        train_step(inputs[:-1])
        # train reply for 1->2, 3->4, 5->6
        train_step(inputs[1:])

        if (step + 1) % logging_step == 0:
            with summary_writer.as_default():
                tf.summary.scalar(
                    'train_loss', train_loss_metric.result(), step=step)
                tf.summary.scalar(
                    'train_accuracy', train_accuracy_metric.result(), step=step)

            logger.info(f'{step+1} / {total_steps}: '
                        f'Loss: {train_loss_metric.result():.3f}, '
                        f'Accuracy: {train_accuracy_metric.result():.3f}')

            train_loss_metric.reset_states()
            train_accuracy_metric.reset_states()

        if (step + 1) % dataloader.steps_per_epoch == 0:
            ckpt_path = ckpt_manager.save()
            logger.info(f'Save checkpoint to {ckpt_path}')
            logger.info(f'Time taken for 1 epoch: '
                        f'{time.time() - start:.3f} secs\n')
            start = time.time()


def build_loss_fn(label_smoothing, vocab_size):
    loss_fn = functools.partial(metrics.transformer_loss,
                                smoothing=label_smoothing,
                                vocab_size=vocab_size)
    return loss_fn


def main(params):
    # Print params
    for k, v in params.items():
        logger.info(f'{k}: {v}')

    model = MemoryTransformer(params)
    loss_fn = build_loss_fn(
        label_smoothing=params['label_smoothing'],
        vocab_size=params['vocab_size'])
    lr_schedule = optimizer.LearningRateSchedule(
        params["learning_rate"], params["hidden_size"],
        params["learning_rate_warmup_steps"])
    opt = tf.keras.optimizers.Adam(
        lr_schedule,
        params["optimizer_adam_beta1"],
        params["optimizer_adam_beta2"],
        epsilon=params["optimizer_adam_epsilon"])

    logger.info('Load data')
    dataloader = DataLoader(
        len_conversations=params['len_conversations'],
        max_token_length=params['max_token_length'],
        dataset_dir=params['dataset_dir'],
        batch_size=params['batch_size'])

    train(epochs=params['epochs'],
          model=model, loss_fn=loss_fn, opt=opt,
          dataloader=dataloader, log_dir=params['log_dir'],
          hidden_size=params['hidden_size'])

    export_path = os.path.join(params['log_dir'], 'saved_model')
    export(model=model, export_path=export_path,
           hidden_size=params['hidden_size'])
    logger.info(f'Export model to {export_path}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='dataset')
    parser.add_argument('--log_dir', type=str, default='log_dir')
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--len_conversations', type=int, default=7)
    parser.add_argument('--max_token_length', type=int, default=64)
    args = parser.parse_args()

    params = {
        'vocab_size': VOCAB_SIZE,
        'hidden_size': 512,
        'num_hidden_layers': 6,
        'num_heads': 8,
        'filter_size': 2048,

        'layer_postprocess_dropout': 0.1,
        'attention_dropout': 0.1,
        'relu_dropout': 0.1,

        'label_smoothing': 0.1,
        'learning_rate': 2.0,
        'learning_rate_decay_rate': 1.0,
        'learning_rate_warmup_steps': 16000,

        'optimizer_adam_beta1': 0.9,
        'optimizer_adam_beta2': 0.997,
        'optimizer_adam_epsilon': 1e-09,

        'extra_decode_length': 50,
        'beam_size': 4,
        'alpha': 0.6,

        'padded_decode': False,
        'dtype': tf.float32
    }

    params.update(vars(args))
    main(params)
