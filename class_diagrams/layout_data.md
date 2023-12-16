@startuml
class FakeBoolFlag {
  - value: bool
}

class NamedTuple

class LayoutExample {
  - total_nodes: tf.Tensor
  - total_edges: tf.Tensor
  - total_configs: tf.Tensor
  - total_config_nodes: tf.Tensor
  - node_features: tf.Tensor
  - node_ops: tf.Tensor
  - edges: tf.Tensor
  - node_config_features: tf.Tensor
  - config_runtimes: tf.Tensor
  - argsort_config_runtimes: tf.Tensor
  - graph_id: tf.Tensor
  - node_config_ids: tf.Tensor
  - node_splits: tf.Tensor
  + to_graph_tensor(config_samples: int, max_nodes: int): tfgnn.GraphTensor
}

class NpzDatasetPartition {
  - _data_dict: dict[str, list[np.ndarray]]
  - _num_edges: list[int]
  - _num_configs: list[int]
  - _num_nodes: list[int]
  - _num_config_nodes: list[int]
  - _num_node_splits: list[int]
  - node_feat: 'tf.Tensor | None'
  - node_opcode: 'tf.Tensor | None'
  - edge_index: 'tf.Tensor | None'
  - config_runtime: 'tf.Tensor | None'
  - argsort_config_runtime: tf.Tensor|None
  - graph_id: 'tf.Tensor | None'
  - node_config_feat: 'tf.Tensor | None'
  - edge_ranges: 'tf.Tensor | None'
  - node_ranges: 'tf.Tensor | None'
  - config_ranges: 'tf.Tensor | None'
  - config_node_ranges: 'tf.Tensor | None'
  - flat_config_ranges: 'tf.Tensor | None'
  - node_split_ranges: 'tf.Tensor | None'
  - node_splits: 'tf.Tensor | None'
  - node_config_ids: 'tf.Tensor | None'
  + save_to_file(cache_file: str): void
  + load_from_file(cache_file: str): void
  + add_npz_file(graph_id: str, npz_file: np.lib.npyio.NpzFile, min_configs: int, max_configs: int): void
  + finalize(): void
  + get_item(index: int): LayoutExample
  + get_graph_tensors_dataset(config_samples: int, max_nodes: int): tf.data.Dataset
  + iter_graph_tensors(): void
}

class NpzDataset {
  - train: NpzDatasetPartition
  - validation: NpzDatasetPartition
  - test: NpzDatasetPartition
  + num_ops: int
  + normalize(): void
  - _get_normalizer(feature_matrix): tuple[tf.Tensor, tf.Tensor, tf.Tensor]
  - _apply_normalizer(feature_matrix, used_columns, min_feat, max_feat): tf.Tensor
}

class TensorFlowGNN {
  <<metaclass>>
}

class tf {
  <<stereotype>> TensorFlow
}

FakeBoolFlag --|> NamedTuple
LayoutExample --|> NamedTuple
NpzDatasetPartition --|> NamedTuple
NpzDataset --|> NamedTuple
LayoutExample --|> TensorFlowGNN
NpzDatasetPartition --|> TensorFlowGNN
@enduml

