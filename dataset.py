from torch.utils.data import Dataset
from collections import defaultdict 
import igraph as ig
import random
import torch
import numpy as np

class Graph: 
    def __init__(self,vertices): 
        self.graph = defaultdict(list) 
        self.V = vertices

    def addEdge(self,u,v): 
        self.graph[u].append(v) 
  
    def topologicalSortUtil(self,v,visited,stack): 
        visited[v] = True
        order = self.graph[v]
        random.shuffle(order)
        for i in order: 
            if visited[i] == False: 
                self.topologicalSortUtil(i,visited,stack) 
        stack.insert(0,v) 
  
    def topologicalSort(self): 
        visited = [False]*self.V 
        stack =[] 
        order = list(range(self.V))
        random.shuffle(order)
        for i in order: 
            if visited[i] == False: 
                self.topologicalSortUtil(i,visited,stack) 
        return stack 

import globalvar as gl

class diffusion_dataset(Dataset):
    
    def __init__(self,data,args):
        self.data = data
        self.args = args
            
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        g = self.data[index]

        event_type_list = g.vs['type']
        topo_g= Graph(len(g.vs)) 
        for i,j in g.get_edgelist():
            topo_g.addEdge(i,j)
        order = topo_g.topologicalSort()
        old2new = dict()
        for i,old in enumerate(order):
            old2new[old] = i
        
        new_g = ig.Graph(len(g.vs),directed=True)
        new_event_type = []
        for i in range(len(order)):
            new_event_type.append(event_type_list[order[i]])
        new_g.vs["type"] = new_event_type

        for i,j in g.get_edgelist():
            new_g.add_edge(old2new[i],old2new[j])

        g = new_g

        random_data = gl.get_value('random_data')
        if len(random_data[index])<self.args.random_num:
            random_data[index].append(g)
            gl.set_value('random_data',random_data)
        else:
            g = random_data[index][random.randint(0,self.args.random_num-1)]
        node_label = g.vs['type']
        attention_mask = [1 for _ in range(len(node_label))] + [0 for _ in range(self.args.max_n-len(node_label))]
        node_label += [self.args.PAD_TYPE for _ in range(self.args.max_n-len(node_label))]
        adj = g.get_adjacency().data

         # Laplacian
        A = torch.tensor(g.get_adjacency().data).float() + torch.tensor(g.get_adjacency().data).T
        N = torch.diag(torch.tensor(np.array(g.degree())).float() ** -0.5)
        L = torch.eye(torch.tensor(len(g.vs))) - torch.mm(torch.mm(N,A),N) # N * A * N

        # Eigenvectors with numpy
        EigVal, EigVec = np.linalg.eig(np.array(L))
        idx = EigVal.argsort() # increasing order
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
        lap_pos_enc = torch.from_numpy(EigVec[:,1:self.args.pos_enc_dim+1]).float() 
        lap_pos_enc = torch.cat([lap_pos_enc,torch.zeros((self.args.max_n-len(g.vs),self.args.pos_enc_dim))],dim=0)    
        edge_matrix = torch.zeros((self.args.max_n,self.args.max_n))
        adj = torch.tensor(adj)
        edge_matrix[0:len(adj),0:len(adj)] = adj
        # g_matrix = torch.cat([node_matrix,edge_matrix],dim=1)
        # node_label = list(range(0,50,1))
        # attention_mask = [1 for _ in range(50)]

        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        weight_mask = edge_matrix.view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0)) 
        weight_tensor[weight_mask] = pos_weight
        pad_mask = torch.zeros((self.args.max_n,self.args.max_n))
        pad_mask[0:len(adj),0:len(adj)] = torch.tensor(1)
        pad_mask = pad_mask.view(-1) == 0
        weight_tensor[pad_mask] = 0
        return [torch.tensor(node_label),torch.tensor(attention_mask),edge_matrix,lap_pos_enc,weight_tensor]

        