import json
import os
import sys
from math import ceil
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import Mnist, Cifar10
from utils.visualize import show_filter, show_image


class Solver(object):
    def __init__(self, dataset='mnist', model='simplecnn', **kwargs):
        if dataset == 'mnist':
            self.dataset = Mnist()
        elif dataset == 'cifar10':
            self.dataset = Cifar10()
        else:
            raise NotImplementedError

        self.model_name = model
        self.hyper_mode = kwargs.pop('hyper_mode', True)

        x_shape = getattr(self.dataset, 'x_train').shape
        self.x_dim = kwargs.pop('x_dim', int(x_shape[1]))
        self.c_dim = kwargs.pop('c_dim', int(x_shape[3]))
        self.num_classes = kwargs.pop('num_classes', int(self.dataset.num_classes))
        self.batch_size = kwargs.pop('batch_size', 1024)
        self.max_epoch = kwargs.pop('max_epoch', 50)
        self.learning_rate = kwargs.pop('learning_rate', 0.0005)
        self.lr_decay = kwargs.pop('lr_decay', 0.99)
        self.grad_clip = kwargs.pop('grad_clip', 100.0)
        self.optimize_method = kwargs.pop('optimizer', 'adam')
        self.logpath = kwargs.pop('logpath', 'log')
        self.val_split = kwargs.pop('val_split', 0.1)
        self.save_dir = kwargs.pop('save_dir', 'checkpoints')
        self.save_best_only = kwargs.pop('save_best_only', False)
        self.resume = kwargs.pop('resume', False)
        self.eval_only = kwargs.pop('eval_only', False)
        self.seed = kwargs.pop('seed', 42)
        self.show_sample = kwargs.pop('show_sample', False)
        self.show_filters = kwargs.pop('show_filters', False)

        self._validate_config()
        self._prepare_output_dirs()
        self._prepare_data_splits()
        self.metric_name = 'val_acc' if self.x_val is not None else 'test_acc'
        self.model = self._create_model()
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE,
        )
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=self.n_iterations,
            decay_rate=self.lr_decay,
        )
        if self.optimize_method == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        else:
            raise NotImplementedError
        self.epoch_var = tf.Variable(0, trainable=False, dtype=tf.int64, name='epoch')
        self.checkpoint = tf.train.Checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=self.epoch_var,
        )
        self.latest_manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=self.latest_dir,
            max_to_keep=1,
            checkpoint_name='ckpt',
        )
        self.best_manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=self.best_dir,
            max_to_keep=1,
            checkpoint_name='ckpt',
        )
        self.history_manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=self.history_dir,
            max_to_keep=5,
            checkpoint_name='ckpt',
        )

    def _validate_config(self):
        if not 0.0 <= self.val_split < 1.0:
            raise ValueError('val_split must be in the range [0.0, 1.0).')
        if self.batch_size <= 0:
            raise ValueError('batch_size must be positive.')
        if self.max_epoch <= 0 and not self.eval_only:
            raise ValueError('epochs must be positive unless running with --eval-only.')

    def _prepare_output_dirs(self):
        os.makedirs(self.logpath, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)

        self.latest_dir = os.path.join(self.save_dir, 'latest')
        self.best_dir = os.path.join(self.save_dir, 'best')
        self.history_dir = os.path.join(self.save_dir, 'history')
        for directory in (self.latest_dir, self.best_dir, self.history_dir):
            os.makedirs(directory, exist_ok=True)

        self.state_path = os.path.join(self.save_dir, 'training_state.json')

    def _prepare_data_splits(self):
        x_train = self.dataset.x_train
        y_train = self.dataset.y_train
        self.x_test = self.dataset.x_test
        self.y_test = self.dataset.y_test

        if self.val_split > 0.0:
            val_size = int(len(x_train) * self.val_split)
            val_size = max(1, val_size)
            if val_size >= len(x_train):
                raise ValueError('val_split leaves no samples for the training split.')

            rng = np.random.RandomState(self.seed)
            permutation = rng.permutation(len(x_train))
            val_indices = permutation[:val_size]
            train_indices = permutation[val_size:]
            self.x_val = x_train[val_indices]
            self.y_val = y_train[val_indices]
            self.x_train = x_train[train_indices]
            self.y_train = y_train[train_indices]
        else:
            self.x_train = x_train
            self.y_train = y_train
            self.x_val = None
            self.y_val = None

        self.n_iterations = max(1, int(ceil(len(self.x_train) / float(self.batch_size))))

    def _create_model(self):
        if self.model_name == 'simplecnn':
            from model.simple_cnn import SimpleCNN

            model = SimpleCNN(num_classes=self.num_classes, hyper_mode=self.hyper_mode)
        elif self.model_name == 'resnet50':
            from model.resnet import Resnet50

            model = Resnet50(num_classes=self.num_classes, hyper_mode=self.hyper_mode).build_model()
        else:
            raise NotImplementedError

        dummy_inputs = tf.zeros((1, self.x_dim, self.x_dim, self.c_dim), dtype=tf.float32)
        model(dummy_inputs, training=False)
        return model

    def _visualize_sample(self):
        sample = self.x_train[0]
        show_image(sample)

    def _get_first_conv_kernel(self):
        candidate_layers = [
            getattr(self.model, 'conv1', None),
            getattr(self.model, 'stem_conv', None),
        ]
        for layer in candidate_layers:
            kernel = getattr(layer, 'kernel', None)
            if kernel is not None:
                return kernel.numpy()
        return None

    def _visualize_filters(self):
        kernel = self._get_first_conv_kernel()
        if kernel is None:
            print('Filter visualization is not available for the current model.')
            return
        show_filter(kernel)

    def _make_dataset(self, x, y, training=False):
        if x is None or y is None:
            return None

        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        if training:
            dataset = dataset.shuffle(buffer_size=len(x), seed=self.seed, reshuffle_each_iteration=True)
        return dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def _compute_loss(self, labels, logits):
        losses = self.loss_fn(labels, logits)
        return tf.reduce_mean(losses), losses

    def _train_step(self, batch_x, batch_y):
        with tf.GradientTape() as tape:
            logits = self.model(batch_x, training=True)
            loss, _ = self._compute_loss(batch_y, logits)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
        gradient_pairs = [
            (gradient, variable)
            for gradient, variable in zip(gradients, self.model.trainable_variables)
            if gradient is not None
        ]
        self.optimizer.apply_gradients(gradient_pairs)
        return logits

    def _current_learning_rate(self):
        learning_rate = self.optimizer.learning_rate
        if callable(learning_rate):
            learning_rate = learning_rate(self.optimizer.iterations)
        return float(tf.keras.backend.get_value(learning_rate))

    def _find_checkpoint(self):
        return self.latest_manager.latest_checkpoint or self.best_manager.latest_checkpoint

    def _restore_checkpoint(self):
        checkpoint = self._find_checkpoint()
        if checkpoint is None:
            raise ValueError('No checkpoint found. Train once before using --resume or --eval-only.')
        status = self.checkpoint.restore(checkpoint)
        status.expect_partial()
        return checkpoint

    def _save_checkpoint(self, epoch, is_best):
        self.epoch_var.assign(epoch)
        self.latest_manager.save(checkpoint_number=epoch)
        if not self.save_best_only:
            self.history_manager.save(checkpoint_number=epoch)
        if is_best:
            self.best_manager.save(checkpoint_number=epoch)

    def _load_training_state(self):
        if not os.path.exists(self.state_path):
            return {
                'completed_epochs': 0,
                'best_epoch': 0,
                'best_metric': None,
                'metric_name': self.metric_name,
            }
        with open(self.state_path, 'r') as state_file:
            state = json.load(state_file)
        state.setdefault('completed_epochs', 0)
        state.setdefault('best_epoch', 0)
        state.setdefault('best_metric', None)
        state.setdefault('metric_name', self.metric_name)
        return state

    def _save_training_state(self, completed_epochs, best_epoch, best_metric, global_step):
        state = {
            'completed_epochs': int(completed_epochs),
            'best_epoch': int(best_epoch),
            'best_metric': None if best_metric is None else float(best_metric),
            'metric_name': self.metric_name,
            'global_step': int(global_step),
        }
        with open(self.state_path, 'w') as state_file:
            json.dump(state, state_file, indent=2, sort_keys=True)

    def _train_epoch(self):
        train_dataset = self._make_dataset(self.x_train, self.y_train, training=True)
        for batch_x, batch_y in train_dataset:
            self._train_step(batch_x, batch_y)

    def _write_scalar_summary(self, summary_writer, tag, value, step):
        if value is None:
            return
        with summary_writer.as_default():
            tf.summary.scalar(tag, value, step=step)

    def _log_epoch_metrics(self, summary_writer, epoch, metrics, learning_rate, best_metric):
        self._write_scalar_summary(summary_writer, 'train/loss', metrics['train'][0], epoch)
        self._write_scalar_summary(summary_writer, 'train/accuracy', metrics['train'][1], epoch)
        self._write_scalar_summary(summary_writer, 'val/loss', metrics['val'][0], epoch)
        self._write_scalar_summary(summary_writer, 'val/accuracy', metrics['val'][1], epoch)
        self._write_scalar_summary(summary_writer, 'test/loss', metrics['test'][0], epoch)
        self._write_scalar_summary(summary_writer, 'test/accuracy', metrics['test'][1], epoch)
        self._write_scalar_summary(summary_writer, 'training/learning_rate', learning_rate, epoch)
        self._write_scalar_summary(summary_writer, 'training/best_metric', best_metric, epoch)
        summary_writer.flush()

    def _print_epoch_metrics(self, prefix, epoch, metrics, learning_rate, is_best, best_metric):
        parts = [
            '%s %3d:' % (prefix, epoch),
            'train loss %.6f' % metrics['train'][0],
            'train acc %.6f' % metrics['train'][1],
        ]
        if metrics['val'][0] is not None:
            parts.extend([
                'val loss %.6f' % metrics['val'][0],
                'val acc %.6f' % metrics['val'][1],
            ])
        parts.extend([
            'test loss %.6f' % metrics['test'][0],
            'test acc %.6f' % metrics['test'][1],
        ])
        if learning_rate is not None:
            parts.append('lr %.6f' % learning_rate)
        if best_metric is not None:
            parts.append('best %s %.6f' % (self.metric_name, best_metric))
        if is_best:
            parts.append('[best]')
        print(', '.join(parts))

    def train(self):
        state = self._load_training_state()
        start_epoch = 0
        best_metric = state['best_metric']
        best_epoch = state['best_epoch']

        if self.show_sample:
            self._visualize_sample()

        if self.resume or self.eval_only:
            checkpoint = self._restore_checkpoint()
            print('Restored checkpoint from %s' % checkpoint)
            state = self._load_training_state()
            start_epoch = state['completed_epochs']
            best_metric = state['best_metric']
            best_epoch = state['best_epoch']

        if self.eval_only:
            metrics = {
                'train': self.evaluate_in_batch(self.x_train, self.y_train),
                'val': self.evaluate_in_batch(self.x_val, self.y_val),
                'test': self.evaluate_in_batch(self.x_test, self.y_test),
            }
            self._print_epoch_metrics('Eval', start_epoch, metrics, None, False, best_metric)
            if self.show_filters:
                self._visualize_filters()
            return

        if start_epoch >= self.max_epoch:
            print('Training already completed up to epoch %d.' % start_epoch)
            return

        summary_writer = tf.summary.create_file_writer(self.logpath)
        for epoch in range(start_epoch, self.max_epoch):
            epoch_id = epoch + 1
            self._train_epoch()
            metrics = {
                'train': self.evaluate_in_batch(self.x_train, self.y_train),
                'val': self.evaluate_in_batch(self.x_val, self.y_val),
                'test': self.evaluate_in_batch(self.x_test, self.y_test),
            }
            learning_rate = self._current_learning_rate()
            selection_metric = metrics['val'][1] if metrics['val'][1] is not None else metrics['test'][1]
            is_best = best_metric is None or selection_metric > best_metric
            if is_best:
                best_metric = selection_metric
                best_epoch = epoch_id

            self._save_checkpoint(epoch_id, is_best)
            self._save_training_state(
                epoch_id,
                best_epoch,
                best_metric,
                int(self.optimizer.iterations.numpy()),
            )
            self._log_epoch_metrics(summary_writer, epoch_id, metrics, learning_rate, best_metric)
            self._print_epoch_metrics('Epoch', epoch_id, metrics, learning_rate, is_best, best_metric)

        print('Best %s %.6f at epoch %d' % (self.metric_name, best_metric, best_epoch))
        if self.show_filters:
            self._visualize_filters()

    def evaluate_in_batch(self, x, y):
        if x is None or y is None:
            return None, None

        dataset = self._make_dataset(x, y, training=False)
        loss_metric = tf.keras.metrics.Mean()
        acc_metric = tf.keras.metrics.CategoricalAccuracy()

        for batch_x, batch_y in dataset:
            logits = self.model(batch_x, training=False)
            _, per_example_losses = self._compute_loss(batch_y, logits)
            probabilities = tf.nn.softmax(logits)
            loss_metric.update_state(per_example_losses)
            acc_metric.update_state(batch_y, probabilities)
        return float(loss_metric.result().numpy()), float(acc_metric.result().numpy())


if __name__ == '__main__':
    solver = Solver(model='simplecnn', dataset='mnist', max_epoch=10)
    solver.train()
