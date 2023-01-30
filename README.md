# graph-transformer

An unofficial implementation of Graph Transformer (Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification) - IJCAI 2021

## Installation

```bash
pip install graph-transformer
```

GraphTransformerModel(512, 512, 2, 8, False)
## Usage
```python
from graph_transformer import GraphTransformerModel

model = GraphTransformerModel(
        node_dim = 512,
        edge_dim = 512,
        num_blocks = 3, # number of graph transformer blocks
        num_heads = 8,
        last_average=True, # wether to average or concatenation at the last block
        model_dim=None # if None, node_dim will be used as the dimension of the graph transformer block
)

nodes = torch.randn(1, 128, 512)
edges = torch.randn(1, 128, 128, 512)
adjacency = torch.ones(1, 128, 128)

nodes = model(nodes, edges, adjacency)
```
