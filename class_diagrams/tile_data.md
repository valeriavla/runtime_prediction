@startuml
class FakeBoolFlag {
  - value: bool
  + _TOY_DATA: FakeBoolFlag
}

class TileExample {
  - node_features: tf.Tensor
  - node_ops: tf.Tensor
  - edges: tf.Tensor
  - config_features: tf.Tensor
  - config_runtimes: tf.Tensor
  - config_runtime_normalizers: tf.Tensor
  - tile_id: tf.Tensor
  - total_nodes: tf.Tensor
  - total_edges: tf.Tensor
  - total_configs: tf.Tensor
  + to_graph_tensor(config_samples: int = -1, normalize_runtimes: bool = True): tfgnn.GraphTensor
}

class NpzDatasetPartition {
  - _data_dict: dict[str, list[np.ndarray]]
  - _num_edges: list[int]
  - _num_configs: list[int]
  - _num_nodes: list[int]
  - node_feat: 'tf.Tensor | None'
  - node_opcode: 'tf.Tensor | None'
  - edge_index: 'tf.Tensor | None'
  - config_feat: 'tf.Tensor | None'
  - config_runtime: 'tf.Tensor | None'
  - config_runtime_normalizers: 'tf.Tensor | None'
  - tile_id: 'tf.Tensor | None'
  - edge_ranges: 'tf.Tensor | None'
  - node_ranges: 'tf.Tensor | None'
  - config_ranges: 'tf.Tensor | None'
  + save_to_file(cache_file: str): void
  + load_from_file(cache_file: str): void
  + add_npz_file(tile_id: str, npz_file: np.lib.npyio.NpzFile, min_configs: int = 2): void
  + finalize(): void
  + get_item(index: int): TileExample
  + get_graph_tensors_dataset(config_samples: int = -1): tf.data.Dataset
}

class NpzDataset {
  - train: NpzDatasetPartition
  - validation: NpzDatasetPartition
  - test: NpzDatasetPartition
  + num_ops: int
  + normalize(): void
}

@enduml
