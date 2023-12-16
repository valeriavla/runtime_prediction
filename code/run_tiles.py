# Install standard modules

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_gnn as tfgnn
import tensorflow_ranking as tfr
import argparse


from tpu_graphs.baselines.layout import data_kaggle as tile_data
from tpu_graphs.baselines.tiles import data_kaggle_tile as tile_data
import tpu_graphs.baselines.tiles.implicit as implicit

import tqdm
import collections
import functools
import hashlib
import io
import os
from typing import NamedTuple

from absl import flags
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
import tqdm

"""
baseline models and functions from:
https://kaggle.com/competitions/predict-ai-model-runtime
"""

class _OpEmbedding(tf.keras.Model):
  """Embeds GraphTensor.node_sets['op']['op'] nodes into feature 'op_e'."""

  def __init__(self, num_ops: int, embed_d: int, l2reg: float = 1e-4):
    super().__init__()
    self.embedding_layer = tf.keras.layers.Embedding(
        num_ops, embed_d, activity_regularizer=tf.keras.regularizers.l2(l2reg))

  def call(
      self, graph: tfgnn.GraphTensor,
      training: bool = False) -> tfgnn.GraphTensor:
    op_features = dict(graph.node_sets['op'].features)
    op_features['op_e'] = self.embedding_layer(
        tf.cast(graph.node_sets['op']['op'], tf.int32))
    return graph.replace_features(node_sets={'op': op_features})


def pair_layout_graph_with_label(graph: tfgnn.GraphTensor):
    """Extracts label from graph (`tfgnn.GraphTensor`) and returns a pair of `(graph, label)`"""
    # Return runtimes divded over large number: only ranking is required. The
    # runtimes are in the 100K range
    label = tf.cast(graph.node_sets['g']['runtimes'], tf.float32) / 1e7
    return graph, label


class ResModel(tf.keras.Model):
    """GNN with residual connections."""

    def __init__(
        self, num_configs: int, num_ops: int, op_embed_dim: int = 32,
        num_gnns: int = 2, mlp_layers: int = 2,
        hidden_activation: str = 'leaky_relu',
        hidden_dim: int = 32, reduction: str = 'sum'):
        super().__init__()
        self._num_configs = num_configs
        self._num_ops = num_ops
        self._op_embedding = _OpEmbedding(num_ops, op_embed_dim)
        self._prenet = _mlp([hidden_dim] * mlp_layers, hidden_activation)
        self._gc_layers = []
        for _ in range(num_gnns):
            self._gc_layers.append(_mlp([hidden_dim] * mlp_layers, hidden_activation))
        self._postnet = _mlp([hidden_dim, 1], hidden_activation, use_bias=False)

    def call(self, graph: tfgnn.GraphTensor, training: bool = False):
        del training
        return self.forward(graph, self._num_configs)

    def _node_level_forward(
        self, node_features: tf.Tensor,
        config_features: tf.Tensor,
        graph: tfgnn.GraphTensor, num_configs: int,
        edgeset_prefix='') -> tf.Tensor:
        """implements the full computation within a GNN layer:
        obtains adjacency Matrices and normalizes them, 
        transforms and normalizes nodes and configuration.
        applies the Pre-processing MLP and performs the Graph Convolution Operation.
        """
    
        adj_op_op = implicit.AdjacencyMultiplier(
            graph, edgeset_prefix+'feed')  # op->op
        adj_config = implicit.AdjacencyMultiplier(
            graph, edgeset_prefix+'config')  # nconfig->op

        adj_op_op_hat = (adj_op_op + adj_op_op.transpose()).add_eye()
        adj_op_op_hat = adj_op_op_hat.normalize_symmetric()

        x = node_features

        x = tf.stack([x] * num_configs, axis=1)
        config_features = 100 * (adj_config @ config_features)
        x = tf.concat([config_features, x], axis=-1)
        x = self._prenet(x)
        x = tf.nn.leaky_relu(x)

        for layer in self._gc_layers:
            y = x
            y = tf.concat([config_features, y], axis=-1)
            y = tf.nn.leaky_relu(layer(adj_op_op_hat @ y))
            x += y
        return x

    def forward(
        self, graph: tfgnn.GraphTensor, num_configs: int,
        backprop=True) -> tf.Tensor:
        """
        Overall forward pass within the embedding layer,
        the node-level forward pass (_node_level_forward),
        and the final global pooling and post-processing stages.
        """
        graph = self._op_embedding(graph)

        config_features = graph.node_sets['nconfig']['feats']
        node_features = tf.concat([
            graph.node_sets['op']['feats'],
            graph.node_sets['op']['op_e']
        ], axis=-1)

        x_full = self._node_level_forward(
            node_features=tf.stop_gradient(node_features),
            config_features=tf.stop_gradient(config_features),
            graph=graph, num_configs=num_configs)

        if backprop:
            x_backprop = self._node_level_forward(
                node_features=node_features,
                config_features=config_features,
                graph=graph, num_configs=num_configs, edgeset_prefix='sampled_')

            is_selected = graph.node_sets['op']['selected']
            # Need to expand twice as `is_selected` is a vector (num_nodes) but
            # x_{backprop, full} are 3D tensors (num_nodes, num_configs, num_feats).
            is_selected = tf.expand_dims(is_selected, axis=-1)
            is_selected = tf.expand_dims(is_selected, axis=-1)
            x = tf.where(is_selected, x_backprop, x_full)
        else:
            x = x_full

        adj_config = implicit.AdjacencyMultiplier(graph, 'config')

        # Features for configurable nodes.
        config_feats = (adj_config.transpose() @ x)

        # Global pooling
        adj_pool_op_sum = implicit.AdjacencyMultiplier(graph, 'g_op').transpose()
        adj_pool_op_mean = adj_pool_op_sum.normalize_right()
        adj_pool_config_sum = implicit.AdjacencyMultiplier(
            graph, 'g_config').transpose()
        x = self._postnet(tf.concat([
            # (A D^-1) @ Features
            adj_pool_op_mean @ x,
            # l2_normalize( A @ Features )
            tf.nn.l2_normalize(adj_pool_op_sum @ x, axis=-1),
            # l2_normalize( A @ Features )
            tf.nn.l2_normalize(adj_pool_config_sum @ config_feats, axis=-1),
        ], axis=-1))

        x = tf.squeeze(x, -1)

        return x

def _mlp(dims, hidden_activation, l2reg=1e-4, use_bias=True):
  """Helper function for multi-layer perceptron (MLP)."""
  layers = []
  for i, dim in enumerate(dims):
    if i > 0:
      layers.append(tf.keras.layers.Activation(hidden_activation))
    layers.append(tf.keras.layers.Dense(
        dim, kernel_regularizer=tf.keras.regularizers.l2(l2reg),
        use_bias=use_bias))
  return tf.keras.Sequential(layers)

"""
  Layout Training Pipeline
  PARAMETERS
  * Batch Size information
    - BATCH_SIZE: number of graphs per batch
    - CONFIGS_PER_GRAPH: number of configurations (features and target values) per graph
    - MAX_KEEP_NODES: useful for dropout
  * Collection to train on
    - SOURCE: can be 'xla' or 'nlp'
    - SEARCH: can be 'random' or 'default'
  """

"""
CREATE DATASETS FOR TRAINING
"""

def pull_data(CONFIGS_PER_GRAPH, MAX_KEEP_NODES, BATCH_SIZE, tile_data_root_dir):
  tiles_npz_dataset = tile_data.get_npz_dataset(
      tile_data_root_dir,
      min_train_configs=CONFIGS_PER_GRAPH,
      cache_dir='cache'
  )

  tile_train_ds = (
        tiles_npz_dataset.train.get_graph_tensors_dataset(
            config_samples = CONFIGS_PER_GRAPH,
            max_nodes=MAX_KEEP_NODES)
        .shuffle(100, reshuffle_each_iteration=True)
        .batch(BATCH_SIZE, drop_remainder=False)
        .map(tfgnn.GraphTensor.merge_batch_to_components)
        .map(pair_layout_graph_with_label))

  tile_valid_ds = (
        tiles_npz_dataset.validation.get_graph_tensors_dataset(
            config_samples = CONFIGS_PER_GRAPH)
        .batch(BATCH_SIZE, drop_remainder=False)
        .map(tfgnn.GraphTensor.merge_batch_to_components)
        .map(pair_layout_graph_with_label))

  return tiles_npz_dataset, tile_train_ds, tile_valid_ds



def create_model(CONFIGS_PER_GRAPH, tiles_npz_dataset):
  model = ResModel(CONFIGS_PER_GRAPH, tiles_npz_dataset.num_ops)

  loss = tfr.keras.losses.ListMLELoss()  # (temperature=10)
  opt = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=0.5)

  model.compile(loss=loss, optimizer=opt, metrics=[
      tfr.keras.metrics.OPAMetric(name='opa_metric'),
  ])

  return model

def train_model(model, epochs, tile_train_ds, tile_valid_ds):
  best_val_opa = -1  # Tracks best validation OPA
  best_val_at_epoch = -1  # At which epoch.

  for i in range(epochs):
      history = model.fit(
          tile_train_ds, epochs=1, verbose=1, validation_data=tile_valid_ds,
          validation_freq=1)

      train_loss = history.history['loss'][-1]
      train_opa = history.history['opa_metric'][-1]
      val_loss = history.history['val_loss'][-1]
      val_opa = history.history['val_opa_metric'][-1]
      if val_opa > best_val_opa:
          best_val_opa = val_opa
          best_val_at_epoch = i
          best_params = {v.ref: v + 0 for v in model.trainable_variables}
          print(' * [@%i] Validation (NEW BEST): %s' % (i, str(val_opa)))
      elif early_stop > 0 and i - best_val_at_epoch >= early_stop:
        print('[@%i] Best accuracy was attained at epoch %i. Stopping.' % (i, best_val_at_epoch))
        break
  # Restore best parameters.
  print('Restoring parameters corresponding to the best validation OPA.')
  assert best_params is not None
  for v in model.trainable_variables:
      v.assign(best_params[v.ref])

  return model, train_loss, train_opa, val_loss, val_opa, best_params


def run_inference(model, _INFERENCE_CONFIGS_BATCH_SIZE, tiles_npz_dataset):
  print('\n\n   Running inference on test set ...\n\n')
  test_rankings = []

  assert tiles_npz_dataset.test.graph_id is not None
  for graph in tqdm.tqdm(tiles_npz_dataset.test.iter_graph_tensors(),
                        total=tiles_npz_dataset.test.graph_id.shape[-1],
                        desc='Inference'):
      # print(graph)
      num_configs = graph.node_sets['g']['runtimes'].shape[-1]
      # print(num_configs)
      # print(MAX_KEEP_NODES)
      # print("\n\n\n")
      all_scores = []
      for i in tqdm.tqdm(range(0, num_configs, _INFERENCE_CONFIGS_BATCH_SIZE)):
          end_i = min(i + _INFERENCE_CONFIGS_BATCH_SIZE, num_configs)
          # Take a cut of the configs.
          node_set_g = graph.node_sets['g']
          subconfigs_graph = tfgnn.GraphTensor.from_pieces(
              edge_sets=graph.edge_sets,
              node_sets={
                  'op': graph.node_sets['op'],
                  'nconfig': tfgnn.NodeSet.from_fields(
                      sizes=graph.node_sets['nconfig'].sizes,
                      features={
                          'feats': graph.node_sets['nconfig']['feats'][:, i:end_i],
                      }),
                  'g': tfgnn.NodeSet.from_fields(
                      sizes=tf.constant([1]),
                      features={
                          'graph_id': node_set_g['graph_id'],
                          'runtimes': node_set_g['runtimes'][:, i:end_i],
                          'kept_node_ratio': node_set_g['kept_node_ratio'],
                      })
              })
          h = model.forward(subconfigs_graph, num_configs=(end_i - i),
                            backprop=False)
          all_scores.append(h[0])
      all_scores = tf.concat(all_scores, axis=0)
      graph_id = graph.node_sets['g']['graph_id'][0].numpy().decode()
      sorted_indices = tf.strings.join(
          tf.strings.as_string(tf.argsort(all_scores)), ';').numpy().decode()
      test_rankings.append((graph_id, sorted_indices))
  return test_rankings

def write_output(test_rankings, output_csv_filename, SOURCE, SEARCH):
    with tf.io.gfile.GFile(output_csv_filename, 'w') as fout:
        fout.write('ID,TopConfigs\n')
        for graph_id, ranks in test_rankings:
            fout.write(f'layout:{SOURCE}:{SEARCH}:{graph_id},{ranks}\n')
    print('\n\n   ***  Wrote', output_csv_filename, '\n\n')

"""
Function to orchestrate training and testing
on georgetown google-drive and colab environment
"""

def main(source, search, **kwargs):
  # tile_data_ROOT = '/npz/layout'
  tile_data_ROOT = '/content/drive/MyDrive/npz 2/layout'
  SOURCE = source  # Can be "xla" or "nlp"
  SEARCH = search  # Can be "random" or "default"

  tile_data_root_dir = os.path.join(
        os.path.expanduser(tile_data_ROOT), SOURCE, SEARCH)

  # Batch size information.
  # BATCH_SIZE = 10  # Number of graphs per batch.
  # CONFIGS_PER_GRAPH = 2  # Number of configurations (features and target values) per graph.
  # MAX_NUM_CONFIGS = 20 # maximum number of configurations to filter for
  # MAX_KEEP_NODES = 100  # Useful for dropout.
  # MAX_TRAIN_CONFIGS = 20

  BATCH_SIZE = kwargs['batch_size']  # Number of graphs per batch.
  CONFIGS_PER_GRAPH = kwargs['configs_per_graph']  # Number of configurations (features and target values) per graph.
  MAX_NUM_CONFIGS = kwargs['max_num_configs'] # maximum number of configurations to filter for
  MAX_KEEP_NODES = kwargs['max_keep_nodes']  # Useful for dropout.
  MAX_TRAIN_CONFIGS = kwargs['max_train_configs']

  # edges "sampled_config" and "sampled_feed" (or, "con50fig" and "feed")
  early_stop = 5  # If validation OPA did not increase in this many epochs, terminate training.
  best_params = None  # Stores parameters corresponding to best validation OPA, to restore to them after training.
  epochs = 1  # Total number of training epochs.

  # pull the data
  
  tiles_npz_dataset, tile_train_ds, tile_valid_ds = pull_data(CONFIGS_PER_GRAPH, MAX_KEEP_NODES, BATCH_SIZE, tile_data_root_dir)
  model = create_model(CONFIGS_PER_GRAPH, tiles_npz_dataset)
  model, train_loss, train_opa, val_loss, val_opa, best_params = train_model(model, epochs, tile_train_ds, tile_valid_ds)


  _INFERENCE_CONFIGS_BATCH_SIZE = 50
  # _INFERENCE_CONFIGS_BATCH_SIZE = 100

  folder_path = '/content/drive/MyDrive/tpu_graphs/outputcsvs/'
  output_csv_filename = f'inference_layout_{SOURCE}_{SEARCH}.csv'
  output_csv_filename = folder_path + output_csv_filename

  test_rankings = run_inference(model, _INFERENCE_CONFIGS_BATCH_SIZE, tiles_npz_dataset)
  write_output(test_rankings, output_csv_filename, SOURCE, SEARCH)
