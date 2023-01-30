""" This GNN architecture is implemented based on Section 3.1 (Graph Transformer) in:
    Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification.
    https://www.ijcai.org/proceedings/2021/0214.pdf.
    
    In the comments of this code, when "Eq (x)" is mentioned, it refers to the equations in the above paper.
        
    The implementation is referred to this repository (https://github.com/lucidrains/graph-transformer-pytorch),
    but with some modifications by referring to the original paper.
    
"""

import math
import torch
from torch import nn, einsum

List = nn.ModuleList

def softmax(x, adjacency, dim=-1, ):
    """ This calculates softmax based on the given adjacency matrix. 
    """
    means = torch.mean(x, dim, keepdim=True)[0]
    x_exp = torch.exp(x-means) * adjacency
    x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
    x_exp_sum[x_exp_sum==0] = 1.
    
    return x_exp/x_exp_sum

class GatedResidual(nn.Module):
    """ This is the implementation of Eq (5), i.e., gated residual connection between block.
    """
    def __init__(self, dim, only_gate=False):
        super().__init__()
        self.lin_res = nn.Linear(dim, dim)
        self.proj = nn.Sequential(
            nn.Linear(dim * 3, 1, bias = False),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(dim)
        self.non_lin = nn.ReLU()
        self.only_gate = only_gate

    def forward(self, x, res):
        res = self.lin_res(res)
        gate_input = torch.cat((x, res, x - res), dim = -1)
        gate = self.proj(gate_input) # Eq (5), this is beta in the paper
        if self.only_gate: # This is for Eq (6), a case when normalizaton and non linearity is not used.
            return x * gate + res * (1 - gate)
        return self.non_lin(self.norm(x * gate + res * (1 - gate)))

class GraphTransformer(nn.Module):
    """ This is the implementation of Eq (3) and Eq (5), which is the graph transformer block.
    """
    def __init__(
        self,
        in_dim,
        out_dim, # head dim
        num_heads = 8,
        edge_dim = None,
        average = False # This is for Eq (6), a case when average is used instead of concatenation.
    ):
        super().__init__()
        self.out_dim = out_dim

        inner_dim = out_dim * num_heads
        
        self.num_heads = num_heads
        self.average = average

        self.lin_q = nn.Linear(in_dim, inner_dim)
        self.lin_k = nn.Linear(in_dim, inner_dim)
        self.lin_v = nn.Linear(in_dim, inner_dim)
        self.lin_e = nn.Linear(edge_dim, inner_dim)
        
    def forward(self, nodes, edges, adjacency):
        h = self.num_heads
        b = nodes.shape[0]
        n_nodes = nodes.shape[1]

        # Eq (3)
        q = self.lin_q(nodes) # batch x n_nodes x dim -> batch x n_nodes x inner_dim
        k = self.lin_k(nodes) # batch x n_nodes x dim -> batch x n_nodes x inner_dim
        
        # Eq (4)
        v = self.lin_v(nodes) # batch x n_nodes x dim -> batch x n_nodes x inner_dim
        
        # Eq (3)
        e = self.lin_e(edges) # batch x n_nodes x n_nodes x edge_dim
        
        # Split the inner_dim into multiple head, b .. (h d) - > (b h) .. d
        # The attention score later will be computed for each head     
        q =q.view(-1, n_nodes, h, self.out_dim).permute(0,2,1,3).reshape(-1, n_nodes, self.out_dim)
        k =k.view(-1, n_nodes, h, self.out_dim).permute(0,2,1,3).reshape(-1, n_nodes, self.out_dim)
        v =v.view(-1, n_nodes, h, self.out_dim).permute(0,2,1,3).reshape(-1, n_nodes, self.out_dim)
        
        e = e.view(-1, n_nodes,n_nodes, h, self.out_dim).permute(0,3,1,2,4).reshape(-1, n_nodes,n_nodes, self.out_dim)
        
        # Add additional dimension in axis=1 so that it can be added with e.
        # Eg. (batch, 1, n_nodes, out_dim) + (batch, n_nodes, n_nodes, out_dim) 
        k = torch.unsqueeze(k, 1)
        v = torch.unsqueeze(v, 1)

        # Eq (3), addition in the attention score computation
        k = k + e
        
        # Eq (4), addition before concatenation of multi-head
        v = v + e
        
        # Scaled dot-product, before softmax, only <q, k + e> in Eq (3)
        sim = einsum('b i d, b i j d -> b i j', q, k) / math.sqrt(self.out_dim)

        # Softmax computation
        adj = adjacency.repeat_interleave(h, dim=0) # repeat the "adjacency" for h times, so the dimension is the same as "sim"
        attn = softmax(sim, adj, dim=-1)
        
        # Eq (4), multiplication of attention with (v+e), and sum over j (neighbours)
        out = einsum('b i j, b i j d -> b i d', attn, v)
        
        if not self.average: # Eq (4), concatenate multi-head
            out = out.view(-1, h, n_nodes, self.out_dim).permute(0,2,1,3).reshape(-1, n_nodes, h*self.out_dim)
        else: # Eq (6), average multi-head
            out = out.view(-1, h, n_nodes, self.out_dim).permute(0,2,1,3)
            out = torch.mean(out, dim=2)
            
        return out

    
class GraphTransformerModel(nn.Module):
    """ This is the overall architecture of the model.    
    """
    def __init__(
        self,
        node_dim,
        edge_dim,
        num_blocks, # number of graph transformer blocks
        num_heads = 8,
        last_average=False, # wether to average or concatenation at the last block
        model_dim=None # if None, node_dim will be used as the dimension of the graph transformer block
    ):
        super().__init__()
        self.layers = List([])

        # to project the node_dim to model_dim, if model_dim is defined
        self.proj_node_dim = None
        if not model_dim:
            model_dim = node_dim
        else:
            self.proj_node_dim= nn.Linear(node_dim, model_dim)
               
        assert model_dim % num_heads == 0
            
        self.lin_output = nn.Linear(model_dim, 1)

        for i in range(num_blocks):
            if not last_average or i<num_blocks-1:
                self.layers.append(List([
                    GraphTransformer(model_dim, out_dim = int(model_dim/num_heads), edge_dim = edge_dim, num_heads = num_heads),
                    GatedResidual(model_dim)
                ]))
            else:
                self.layers.append(List([
                    GraphTransformer(model_dim, out_dim = model_dim, edge_dim = edge_dim,  num_heads = num_heads, average=True),
                    GatedResidual(model_dim, only_gate=True)
                ]))

    def forward(self, nodes, edges, adjacency):

        if self.proj_node_dim:
            nodes = self.proj_node_dim(nodes)
            
        for trans_block in self.layers:
            trans, trans_residual = trans_block
            nodes = trans_residual(trans(nodes, edges, adjacency), nodes)
                
        return nodes