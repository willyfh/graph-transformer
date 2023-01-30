# Graph Transformer (IJCAI 2021)

An unofficial implementation of Graph Transformer:<br/>
Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification) - IJCAI 2021 > https://www.ijcai.org/proceedings/2021/0214.pdf

This GNN architecture is implemented based on Section 3.1 (Graph Transformer) in the paper.

I implemented the code by referring to [this repository](https://github.com/lucidrains/graph-transformer-pytorch), but with some modifications to match with the original paper.

![image](https://github.com/willyfh/graph-transformer/blob/main/graph-transformer-architecture.png?raw=true)

## Installation

```bash
pip install graph-transformer
```
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
