import torch
import numpy as np

def postprocess_graphs(G,args):
    update_G = []
    for g in G:
        if g.vs['type'][0] == args.START_TYPE and args.END_TYPE ==  g.vs['type'][-1]:
            node2start = g.get_adjacency().data[0]
            for idx,i in enumerate(node2start):
                if i!=0:
                    g.delete_edges(0,idx) 
            end2node = g.get_adjacency().data[-1]
            for idx,i in enumerate(end2node):
                if i!=0:
                    g.delete_edges(idx,len(g.vs['type'])-1)                     
            for i,degree in enumerate(g.degree(mode="in")):
                if i!=0 and i!=len(g.vs['type'])-1 and degree==0:
                    g.add_edge(0,i)
            for i,degree in enumerate(g.degree(mode="out")):
                if i!=0 and i!=len(g.vs['type'])-1 and degree==0:
                    g.add_edge(i,len(g.vs['type'])-1)
        update_G.append(g)
    return update_G

def convert_g2seq_adj_lap(data,args):
    n_num = []
    e_num = []
    examples = []
    for g in data:
        n_num.append(len(g.vs))
        e_num.append(torch.sum(torch.tensor(g.get_adjacency().data)).item())
        node_label = g.vs['type']
        attention_mask = [1 for _ in range(len(node_label))] + [0 for _ in range(args.max_n-len(node_label))]
        node_label += [args.PAD_TYPE for _ in range(args.max_n-len(node_label))]
        adj = g.get_adjacency().data

         # Laplacian
        A = torch.tensor(g.get_adjacency().data).float() + torch.tensor(g.get_adjacency().data).T
        N = torch.diag(torch.tensor(np.array(g.degree())).float() ** -0.5)
        L = torch.eye(torch.tensor(len(g.vs))) - torch.mm(torch.mm(N,A),N) # N * A * N

        # Eigenvectors with numpy
        EigVal, EigVec = np.linalg.eig(np.array(L))
        idx = EigVal.argsort() # increasing order
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
        lap_pos_enc = torch.from_numpy(EigVec[:,1:args.pos_enc_dim+1]).float() 
        lap_pos_enc = torch.cat([lap_pos_enc,torch.zeros((args.max_n-len(g.vs),args.pos_enc_dim))],dim=0)    
        edge_matrix = torch.zeros((args.max_n,args.max_n))
        adj = torch.tensor(adj)
        edge_matrix[0:len(adj),0:len(adj)] = adj

        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        weight_mask = edge_matrix.view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0)) 
        weight_tensor[weight_mask] = pos_weight
        pad_mask = torch.zeros((args.max_n,args.max_n))
        pad_mask[0:len(adj),0:len(adj)] = torch.tensor(1)
        pad_mask = pad_mask.view(-1) == 0
        weight_tensor[pad_mask] = 0
        examples.append([torch.tensor(node_label),torch.tensor(attention_mask),edge_matrix,lap_pos_enc,weight_tensor])
    return examples

# import manifolds
# def tan_proj(emb,c=1):
#     if not torch.is_tensor(emb):
#         emb = torch.from_numpy(emb)
#     manifold = getattr(manifolds, 'PoincareBall')()
#     z_hyp = manifold.expmap0(emb, c)
#     z_hyp = manifold.proj(z_hyp, c) 
#     z_tan = manifold.proj_tan0(z_hyp, c) 
#     return z_tan