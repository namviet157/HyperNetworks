import json
import os
import sys
from math import ceil
from pathlib import Path

import numpy as np
import tensorflow as tf

import model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from my_datasets import Mnist, Cifar10, SVHN, FashionMnist
from utils.visualize import show_filter, show_image


class Solver(object):
    def __init__(self, dataset='mnist', model='simplecnn', **kwargs):
        self.dataset_key = dataset
        if dataset == 'mnist':
            self.dataset = Mnist()
        elif dataset == 'cifar10':
            self.dataset = Cifar10()
        elif dataset == 'fashion_mnist':
            self.dataset = FashionMnist()
        elif dataset == 'svhn':
            self.dataset = SVHN()
        else:
            raise NotImplementedError

        self.model_name = model
        self.hyper_mode = kwargs.pop('hyper_mode', False)

        x_shape = getattr(self.dataset, 'x_train').shape
        self.x_dim = int(x_shape[1])
        self.c_dim = int(x_shape[3])
        self.num_classes = int(self.dataset.num_classes)
        self.batch_size = kwargs.pop('batch_size', 1024)
        self.max_epoch = kwargs.pop('max_epoch', 50)
        self.max_steps = kwargs.pop('max_steps', None)
        self.learning_rate = kwargs.pop('learning_rate', 0.0005)
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.paper_cifar_setup = kwargs.pop(
            'paper_cifar_setup',
            model in ('wrn40_2', 'resnet50'),
        )
        self.grad_clip = kwargs.pop('grad_clip', None)
        self.weight_decay = kwargs.pop(
            'weight_decay',
            5e-6 if self.paper_cifar_setup else 0.0,
        )
        self.label_smoothing = kwargs.pop('label_smoothing', 0.0)
        self.min_learning_rate = kwargs.pop('min_learning_rate', 1e-6)
        self.augment_data = kwargs.pop('augment_data', self.paper_cifar_setup)
        self.augmentation = kwargs.pop(
            'augmentation',
            self._default_augmentation(),
        )
        self.early_stopping_patience = kwargs.pop('early_stopping_patience', None)
        default_optimizer = (
            'adam'
            if self.hyper_mode or not self.paper_cifar_setup
            else 'sgd_nesterov'
        )
        self.optimize_method = kwargs.pop('optimizer', default_optimizer)
        self.lr_schedule_name = kwargs.pop(
            'lr_schedule',
            'paper' if self.paper_cifar_setup else 'exponential',
        )
        self.logpath = kwargs.pop('logpath', 'log')
        self.val_split = kwargs.pop('val_split', 5000)
        self.save_dir = kwargs.pop('save_dir', 'checkpoints')
        self.save_best_only = kwargs.pop('save_best_only', True)
        self.resume = kwargs.pop('resume', False)
        self.eval_only = kwargs.pop('eval_only', False)
        self.seed = kwargs.pop('seed', 42)
        self.show_sample = kwargs.pop('show_sample', False)
        self.show_filters = kwargs.pop('show_filters', False)

        self._validate_config()
        self._prepare_output_dirs()
        self._prepare_data_splits()
        if self.max_steps is not None:
            self.max_epoch = max(
                self.max_epoch,
                int(ceil(float(self.max_steps) / float(self.n_iterations))),
            )
        self.metric_name = 'val_acc' if self.x_val is not None else 'test_acc'
        self.model = self._create_model()
        self._configure_model_layers()
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE,
            label_smoothing=self.label_smoothing,
        )
        self.lr_schedule = self._create_learning_rate_schedule()
        self.optimizer = self._create_optimizer()
        if hasattr(self.optimizer, 'build'):
            self.optimizer.build(self.model.trainable_variables)
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
        if self.val_split < 0:
            raise ValueError('val_split must be a non-negative validation sample count.')
        if self.batch_size <= 0:
            raise ValueError('batch_size must be positive.')
        if self.max_epoch <= 0 and not self.eval_only:
            raise ValueError('max_epoch must be positive unless running with eval_only.')

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

        if self.val_split > 0:
            val_size = int(self.val_split)
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
        elif self.model_name == 'wrn40_2':
            from model.resnet import WideResnet40_2

            model = WideResnet40_2(
                num_classes=self.num_classes, hyper_mode=self.hyper_mode
            ).build_model()
        else:
            raise NotImplementedError

        input_shape = (self.x_dim, self.x_dim, self.c_dim)
        if hasattr(model, 'build_graph'):
            model.build_graph(input_shape).summary()
        else:
            _ = model(tf.zeros((1,) + input_shape), training=False)
            model.summary()

        return model

    def _default_augmentation(self):
        if self.x_dim == 28 and self.c_dim == 1:
            return 'pad_crop'
        if self.x_dim >= 32 and self.c_dim == 3:
            return 'pad_crop_flip'
        return None

    def _configure_model_layers(self):
        track_generated_kernel_l2 = self.weight_decay > 0.0
        for layer in self._iter_model_layers():
            if hasattr(layer, 'track_generated_kernel_l2'):
                layer.track_generated_kernel_l2 = track_generated_kernel_l2

    def _paper_schedule_step_budget(self):
        """Global step budget for piecewise LR (matches max_steps or full epoch loop)."""
        if self.max_steps is not None:
            return max(1, int(self.max_steps))
        return max(1, int(self.max_epoch) * int(self.n_iterations))

    def _create_learning_rate_schedule(self):
        if self.lr_schedule_name == 'paper':
            total_steps = self._paper_schedule_step_budget()
            if self.hyper_mode:
                # Three drops at 25% / 50% / 75% of budget (672k-step paper default).
                boundary_fracs = (0.25, 0.5, 0.75)
                values = [0.002, 0.001, 0.0002, 0.00005]
            else:
                # Four drops at 20% / 40% / 60% / 80% (140k-step WRN baseline default).
                boundary_fracs = (0.2, 0.4, 0.6, 0.8)
                values = [0.1, 0.02, 0.004, 0.0008, 0.00016]
            boundaries = []
            last_b = -1
            cap = max(0, total_steps - 1)
            for frac in boundary_fracs:
                b = int(total_steps * frac)
                b = max(last_b + 1, min(b, cap))
                if b > cap:
                    break
                boundaries.append(b)
                last_b = b
            if not boundaries:
                return float(values[-1])
            values = values[: len(boundaries) + 1]
            return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                boundaries=boundaries,
                values=values,
            )
        if self.lr_schedule_name == 'exponential' and self.lr_decay < 1.0:
            return tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.learning_rate,
                decay_steps=self.n_iterations,
                decay_rate=self.lr_decay,
            )
        return self.learning_rate

    def _create_optimizer(self):
        if self.optimize_method == 'sgd_nesterov':
            return tf.keras.optimizers.SGD(
                learning_rate=self.lr_schedule,
                momentum=0.9,
                nesterov=True,
            )
        if self.optimize_method == 'adam':
            return tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        raise ValueError('Unsupported optimizer: %s' % self.optimize_method)

    def _visualize_sample(self):
        sample = self.x_train[1]
        label = self.y_train[1]
        original_path = os.path.join(self.logpath, 'sample_original.png')
        print('Original training sample:')
        show_image(sample, save_path=original_path)
        print('Saved original training sample to %s' % original_path)
        if not self.augment_data:
            print('Data augmentation is disabled for this run.')
            return
        augmented_sample, _ = self._augment_example(
            tf.convert_to_tensor(sample),
            tf.convert_to_tensor(label),
        )
        augmented_path = os.path.join(self.logpath, 'sample_augmented.png')
        print('Augmented training sample:')
        show_image(augmented_sample.numpy(), save_path=augmented_path)
        print('Saved augmented training sample to %s' % augmented_path)

    def _get_visualizable_kernel(self):
        for layer in self._iter_model_layers():
            generate_kernel = getattr(layer, '_generate_kernel', None)
            if generate_kernel is not None:
                kernel = generate_kernel().numpy()
                return layer.name, 'hyper-generated', kernel

        for layer in self._iter_model_layers():
            kernel = getattr(layer, 'kernel', None)
            if kernel is not None:
                return layer.name, 'trainable', kernel.numpy()
        return None, None, None

    def _preview_kernel(self, kernel, max_input_channels=16, max_output_channels=16):
        if kernel.ndim != 4:
            return kernel
        return kernel[
            :,
            :,
            : min(kernel.shape[2], max_input_channels),
            : min(kernel.shape[3], max_output_channels),
        ]

    def _visualize_filters(self):
        layer_name, kernel_type, kernel = self._get_visualizable_kernel()
        if kernel is None:
            print('Filter visualization is not available for the current model.')
            return
        preview = self._preview_kernel(kernel)
        print(
            'Visualizing %s kernel from layer %s with shape %s.'
            % (kernel_type, layer_name, kernel.shape)
        )
        if preview.shape != kernel.shape:
            print('Showing preview slice with shape %s.' % (preview.shape,))
        filter_path = os.path.join(
            self.logpath,
            'filters_%s_%s.png' % (layer_name, kernel_type.replace('-', '_')),
        )
        show_filter(preview, save_path=filter_path)
        print('Saved filter visualization to %s' % filter_path)

    def _make_dataset(self, x, y, training=False):
        if x is None or y is None:
            return None

        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        if training:
            dataset = dataset.shuffle(buffer_size=len(x), seed=self.seed, reshuffle_each_iteration=True)
            if self.augment_data:
                dataset = dataset.map(self._augment_example, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def _augment_example(self, image, label):
        if self.augmentation in ('mnist', 'pad_crop'):
            image = tf.pad(image, [[1, 1], [1, 1], [0, 0]], mode='CONSTANT')
            image = tf.image.random_crop(image, size=(self.x_dim, self.x_dim, self.c_dim))
        elif self.augmentation in ('cifar', 'pad_crop_flip'):
            image = tf.pad(image, [[4, 4], [4, 4], [0, 0]], mode='REFLECT')
            image = tf.image.random_crop(image, size=(self.x_dim, self.x_dim, self.c_dim))
            image = tf.image.random_flip_left_right(image)
        return image, label

    def _compute_loss(self, labels, logits):
        losses = self.loss_fn(labels, logits)
        return tf.reduce_mean(losses), losses

    def _iter_model_layers(self):
        if hasattr(self.model, '_flatten_layers'):
            yield from self.model._flatten_layers(include_self=False, recursive=True)
            return
        for layer in getattr(self.model, 'layers', []):
            yield layer
            for child in getattr(layer, 'layers', []):
                yield child

    def _regularization_loss(self):
        if self.weight_decay <= 0.0:
            return 0.0
        penalties = [
            tf.nn.l2_loss(variable)
            for variable in self.model.trainable_variables
            if len(variable.shape) > 1 and 'kernel' in variable.name
        ]
        penalties.extend(self.model.losses)
        if not penalties:
            return 0.0
        return self.weight_decay * tf.add_n(penalties)

    @tf.function(reduce_retracing=True)
    def _train_step(self, batch_x, batch_y):
        with tf.GradientTape() as tape:
            logits = self.model(batch_x, training=True)
            data_loss, _ = self._compute_loss(batch_y, logits)
            loss = data_loss + self._regularization_loss()
        gradients = tape.gradient(loss, self.model.trainable_variables)
        if self.grad_clip is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
        gradient_pairs = [
            (gradient, variable)
            for gradient, variable in zip(gradients, self.model.trainable_variables)
            if gradient is not None
        ]
        self.optimizer.apply_gradients(gradient_pairs)
        return logits, data_loss

    @tf.function(reduce_retracing=True)
    def _eval_step(self, batch_x, batch_y):
        logits = self.model(batch_x, training=False)
        _, per_example_losses = self._compute_loss(batch_y, logits)
        probabilities = tf.nn.softmax(logits)
        return per_example_losses, probabilities

    def _current_learning_rate(self):
        learning_rate = self.optimizer.learning_rate
        if callable(learning_rate):
            learning_rate = learning_rate(self.optimizer.iterations)
        return float(tf.keras.backend.get_value(learning_rate))

    def _set_learning_rate(self, learning_rate):
        learning_rate = max(float(learning_rate), self.min_learning_rate)
        current = self.optimizer.learning_rate
        if hasattr(current, 'assign'):
            current.assign(learning_rate)
        else:
            self.optimizer.learning_rate = learning_rate

    def _assert_model_weights_finite(self) -> None:
        for w in self.model.weights:
            if not np.isfinite(w.numpy()).all():
                raise ValueError(
                    'Non-finite weight %s after checkpoint restore.' % w.name
                )

    def _restore_weights_from_path(self, checkpoint_path: str) -> None:
        try:
            self.model.load_weights(checkpoint_path)
        except Exception as exc:
            print(
                'model.load_weights failed (%s); using Checkpoint(model=...).restore.'
                % exc
            )
            tf.train.Checkpoint(model=self.model).restore(
                checkpoint_path
            ).expect_partial()
        self._assert_model_weights_finite()

    def _checkpoint_path_for_eval(self):
        """Prefer ``best/``; if missing (e.g. not trained locally), use ``latest/``."""
        path = self.best_manager.latest_checkpoint
        if path is not None:
            return path, 'best'
        path = self.latest_manager.latest_checkpoint
        if path is not None:
            print(
                'No checkpoint in best/ (%s); loading weights from latest/ instead.'
                % self.best_dir
            )
            return path, 'latest'
        raise ValueError(
            'No checkpoints found under this run. Train first, e.g.:\n'
            '  best:  %s\n'
            '  latest: %s\n'
            'Use val_split > 0 during training so best/ receives the best validation snapshot.'
            % (self.best_dir, self.latest_dir)
        )

    def _restore_best_checkpoint(self, weights_only=False):
        checkpoint_path, source = self._checkpoint_path_for_eval()
        if weights_only:
            self._restore_weights_from_path(checkpoint_path)
            print(
                'Restored eval weights from %s checkpoint: %s'
                % (source, checkpoint_path)
            )
            return checkpoint_path
        status = self.checkpoint.restore(checkpoint_path)
        status.expect_partial()
        return checkpoint_path

    def evaluate_best_checkpoint(self):
        checkpoint = self._restore_best_checkpoint(weights_only=True)
        return {
            'checkpoint': checkpoint,
            'train': self.evaluate_in_batch(self.x_train, self.y_train),
            'val': self.evaluate_in_batch(self.x_val, self.y_val),
            'test': self.evaluate_in_batch(self.x_test, self.y_test),
        }

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

    def _load_existing_best_state(self):
        state = self._load_training_state()
        best_metric = state.get('best_metric')
        best_epoch = state.get('best_epoch', 0)
        metric_name = state.get('metric_name')

        if best_metric is None:
            return None, 0
        if metric_name != self.metric_name:
            print(
                'Ignoring previous best checkpoint because metric changed from %s to %s.'
                % (metric_name, self.metric_name)
            )
            return None, 0
        if self.best_manager.latest_checkpoint is None:
            print(
                'Ignoring previous best metric %.6f because no checkpoint exists under %s.'
                % (best_metric, self.best_dir)
            )
            return None, 0

        print(
            'Found previous best %s %.6f at epoch %d; new checkpoints must improve it.'
            % (self.metric_name, best_metric, best_epoch)
        )
        return float(best_metric), int(best_epoch)

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
        loss_metric = tf.keras.metrics.Mean()
        acc_metric = tf.keras.metrics.CategoricalAccuracy()
        for batch_x, batch_y in train_dataset:
            if self.max_steps is not None and int(self.optimizer.iterations.numpy()) >= self.max_steps:
                break
            logits, loss = self._train_step(batch_x, batch_y)
            loss_metric.update_state(loss)
            acc_metric.update_state(batch_y, tf.nn.softmax(logits))
        return float(loss_metric.result().numpy()), float(acc_metric.result().numpy())

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
            'train loss: %.6f' % metrics['train'][0],
            'train acc: %.6f' % metrics['train'][1],
        ]
        if metrics['val'][0] is not None:
            parts.extend([
                'val loss: %.6f' % metrics['val'][0],
                'val acc: %.6f' % metrics['val'][1],
            ])
        parts.extend([
            'test loss: %.6f' % metrics['test'][0],
            'test acc: %.6f' % metrics['test'][1],
        ])
        if learning_rate is not None:
            parts.append('lr: %.6f' % learning_rate)
        if best_metric is not None:
            parts.append('best %s: %.6f' % (self.metric_name, best_metric))
        if is_best:
            parts.append('[best]')
        print(', '.join(parts))

    def train(self):
        start_epoch = 0
        best_metric = None
        best_epoch = 0

        if self.show_sample:
            self._visualize_sample()

        if self.eval_only:
            self._restore_best_checkpoint(weights_only=True)
            state = self._load_training_state()
            start_epoch = state['completed_epochs']
            best_metric = state['best_metric']
            best_epoch = state['best_epoch']

            metrics = {
                'train': self.evaluate_in_batch(self.x_train, self.y_train),
                'val': self.evaluate_in_batch(self.x_val, self.y_val),
                'test': self.evaluate_in_batch(self.x_test, self.y_test),
            }
            # self._print_epoch_metrics('Eval', start_epoch, metrics, None, False, best_metric)
            if self.show_filters:
                self._visualize_filters()
            return metrics

        if start_epoch >= self.max_epoch:
            print('Training already completed up to epoch %d.' % start_epoch)
            return None

        best_metric, best_epoch = self._load_existing_best_state()
        previous_best_metric = best_metric
        summary_writer = tf.summary.create_file_writer(self.logpath)
        epochs_without_improvement = 0
        for epoch in range(start_epoch, self.max_epoch):
            if self.max_steps is not None and int(self.optimizer.iterations.numpy()) >= self.max_steps:
                break
            epoch_id = epoch + 1
            train_metrics = self._train_epoch()
            metrics = {
                'train': train_metrics,
                'val': self.evaluate_in_batch(self.x_val, self.y_val),
                'test': self.evaluate_in_batch(self.x_test, self.y_test),
            }
            learning_rate = self._current_learning_rate()
            selection_metric = metrics['val'][1] if metrics['val'][1] is not None else metrics['test'][1]
            is_best = best_metric is None or selection_metric > best_metric
            if is_best:
                if previous_best_metric is not None:
                    print(
                        'New best %s %.6f improved previous best %.6f.'
                        % (self.metric_name, selection_metric, previous_best_metric)
                    )
                    previous_best_metric = None
                best_metric = selection_metric
                best_epoch = epoch_id
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            self._save_checkpoint(epoch_id, is_best)
            self._save_training_state(
                epoch_id,
                best_epoch,
                best_metric,
                int(self.optimizer.iterations.numpy()),
            )
            self._log_epoch_metrics(summary_writer, epoch_id, metrics, learning_rate, best_metric)
            self._print_epoch_metrics('Epoch', epoch_id, metrics, learning_rate, is_best, best_metric)
            if (
                self.early_stopping_patience is not None
                and epochs_without_improvement >= self.early_stopping_patience
            ):
                print('Early stopping at epoch %d.' % epoch_id)
                break

        print('Best %s %.6f at epoch %d' % (self.metric_name, best_metric, best_epoch))

        if self.show_filters:
            self._visualize_filters()

        return None

    def evaluate_in_batch(self, x, y):
        if x is None or y is None:
            return None, None

        dataset = self._make_dataset(x, y, training=False)
        loss_metric = tf.keras.metrics.Mean()
        acc_metric = tf.keras.metrics.CategoricalAccuracy()

        for batch_x, batch_y in dataset:
            per_example_losses, probabilities = self._eval_step(batch_x, batch_y)
            loss_metric.update_state(per_example_losses)
            acc_metric.update_state(batch_y, probabilities)
        return float(loss_metric.result().numpy()), float(acc_metric.result().numpy())


if __name__ == '__main__':
    solver = Solver(model='simplecnn', dataset='mnist', max_epoch=10, hyper_mode=True)
    solver.train()
