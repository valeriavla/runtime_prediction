@startuml

class tf.keras.layers.Activation {
  +__init__(activation)
}

class tf.keras.layers.Dense {
  +__init__(units, kernel_regularizer, use_bias)
}

class tf.keras.Sequential {
  +__init__(layers)
}

class tf.keras.layers.Embedding {
  +__init__(input_dim, output_dim, activity_regularizer)
}

class os.path {
  +join(path, *paths)
}

class layout_data {
  +get_npz_dataset(root_dir, min_train_configs, max_train_configs, cache_dir)
}

class tfgnn.GraphTensor {
  +__init__(node_sets, edge_sets)
  +replace_features(node_sets)
}

class implicit.AdjacencyMultiplier {
  +__init__(graph, edgeset_prefix)
  +add_eye()
  +normalize_symmetric()
  +normalize_right()
}

class _OpEmbedding {
  +__init__(num_ops, embed_d, l2reg)
  +call(graph, training)
}

class ResModel {
  +__init__(num_configs, num_ops, op_embed_dim, num_gnns, mlp_layers, hidden_activation, hidden_dim, reduction)
  +call(graph, training)
  +_node_level_forward(node_features, config_features, graph, num_configs, edgeset_prefix)
  +forward(graph, num_configs, backprop)
}

class tfr.keras.losses.ListMLELoss {
  +__init__()
}

class tf.keras.optimizers.Adam {
  +__init__(learning_rate, clipnorm)
}

class tfr.keras.metrics.OPAMetric {
  +__init__(name)
}

class tqdm.tqdm {
  +__init__(iterable, total, desc)
}

class tf.io.gfile.GFile {
  +__init__(filename, mode)
  +write(content)
}

class implicit.AdjacencyMultiplier {
  +__init__(graph, edgeset_prefix)
}

class _INFERENCE_CONFIGS_BATCH_SIZE {
  __init__()
}

class tf.strings {
  +join(strings, separator)
  +as_string(tensor)
}

class tf.argsort {
  +__init__(values)
}

class tf.io.gfile.GFile {
  +__init__(filename, mode)
  +write(content)
}

class tf.strings {
  +as_string(input_tensor, precision, scientific)
}



tf.keras.layers.Activation --|> tf.keras.layers.Layer
tf.keras.layers.Dense --|> tf.keras.layers.Layer
tf.keras.Sequential --|> tf.keras.Model
tf.keras.layers.Embedding --|> tf.keras.layers.Layer
tfgnn.GraphTensor --|> tf.Module
implicit.AdjacencyMultiplier --|>tf.Module
_OpEmbedding --|> tf.keras.Model
ResModel --|>tf.keras.Model
tfr.keras.losses.ListMLELoss --|>tf.losses.Loss
tf.keras.optimizers.Adam --|> tf.optimizers.Optimizer
tfr.keras.metrics.OPAMetric --|> tf.keras.metrics.Metric

@enduml