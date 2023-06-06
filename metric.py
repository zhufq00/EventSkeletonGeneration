from multiprocessing import Pool
from collections import Counter
import torch
# from multiset import Multiset
from tqdm import tqdm
from grakel import Graph
from grakel.kernels import RandomWalkLabeled
import math

    

def randomwalklabeled_single(data):
    g,update_G_true = data
    sp_kernel = RandomWalkLabeled()
    sp_kernel.fit_transform([g])
    k = []
    for g_true in update_G_true:
        k.append(sp_kernel.transform([g_true])[0][0])
    return sum(k)/len(k)

def randomwalklabeled(G,G_true):
    update_G = []
    update_G_true = []
    results = []
    for g in G:
        node_labels = {}
        for i,type in enumerate(g.vs['type']):
            node_labels[i] = type
        update_g  = Graph(initialization_object=g.get_adjacency().data, node_labels=node_labels)
        update_G.append(update_g)
    for g in G_true:
        node_labels = {}
        for i,type in enumerate(g.vs['type']):
            node_labels[i] = type
        update_g  = Graph(initialization_object=g.get_adjacency().data, node_labels=node_labels)
        update_G_true.append(update_g)

    results = []
    with Pool(8) as p:
        for idx,result in enumerate(tqdm(p.imap(randomwalklabeled_single,zip(update_G,[update_G_true for _ in range(len(update_G))])),total=len(update_G))):
            results.append(result)
    index = results.index(max(results))
    results_mv_nan = []
    for result in results:
        if not math.isnan(result):
            results_mv_nan.append(result)
    return sum(results_mv_nan)/len(results_mv_nan),results,index

def randomwalklabeled_old(G,G_true):
    update_G = []
    update_G_true = []
    results = []
    for g in G:
        node_labels = {}
        for i,type in enumerate(g.vs['type']):
            node_labels[i] = type
        update_g  = grakel_Graph(initialization_object=g.get_adjacency().data, node_labels=node_labels)
        update_G.append(update_g)
    for g in G_true:
        node_labels = {}
        for i,type in enumerate(g.vs['type']):
            node_labels[i] = type
        update_g  = grakel_Graph(initialization_object=g.get_adjacency().data, node_labels=node_labels)
        update_G_true.append(update_g)
    for g in tqdm(update_G,total=len(update_G)):
        sp_kernel = RandomWalkLabeled(normalize=True)
        sp_kernel.fit_transform([g])
        k = []
        for g_true in update_G_true:
            k.append(sp_kernel.transform([g_true])[0][0])
        results.append(sum(k)/len(k))
    index = results.index(max(results))
    return sum(results)/len(results),results,index

def event_type_match_F1_single(data):
    g,G_true = data
    sum_F1 = 0
    g_event_type_set = g
    for g_true in G_true: 
        try:
            TP = 0
            FP = 0
            FN = 0
            g_true_event_type_set = set(g_true.vs['type'])
            for type in g_event_type_set:
                if type in g_true_event_type_set:
                    TP+=1
                else:
                    FP+=1
            for type in g_true_event_type_set:
                if type not in g_event_type_set:
                    FN+=1
            pre = TP/(TP+FP)
            rec = TP/(TP+FN)
            F1 = 2*pre*rec/(pre+rec)
        except:
            F1 = 0
        sum_F1+=F1
    return sum_F1/len(G_true)

def event_type_match_F1_fast(G,G_true):
    G_F1_list = []
    with Pool(1) as p:
        for idx,_F1 in enumerate(tqdm(p.imap(event_type_match_F1_single,zip(G,[G_true for _ in range(len(G))])),total=len(G))):
            G_F1_list.append(_F1)
    index = G_F1_list.index(max(G_F1_list))
    return sum(G_F1_list)/len(G_F1_list),G_F1_list,index

def event_type_match_F1(G,G_true):
    G_F1_list = []
    for g in tqdm(G,total=len(G)):
        sum_F1 = 0
        g_event_type_set = set(g.vs['type'])
        for g_true in G_true: 
            try:
                TP = 0
                FP = 0
                FN = 0
                g_true_event_type_set = set(g_true.vs['type'])
                curr_list_true = g_true_event_type_set
                curr_list_pred = g_event_type_set
                intersection = list((Counter(curr_list_true) & Counter(curr_list_pred)).elements())
                # if len(curr_list_pred) == 0 or len(curr_list_true) == 0:
                #     return 0., 0., 0.
                precision = len(intersection) * 1.0 / len(curr_list_pred)
                recall = len(intersection) * 1.0 / len(curr_list_true)
                if precision + recall == 0:
                    f1 = 0.
                else:
                    f1 = 2 * precision * recall / (precision + recall)
                F1 = f1
                # for type in g_event_type_set:
                #     if type in g_true_event_type_set:
                #         TP+=1
                #     else:
                #         FP+=1
                # for type in g_true_event_type_set:
                #     if type not in g_event_type_set:
                #         FN+=1
                # pre = TP/(TP+FP)
                # rec = TP/(TP+FN)
                # F1 = 2*pre*rec/(pre+rec)
            except:
                F1 = 0
            sum_F1+=F1
        G_F1_list.append(sum_F1/len(G_true))
    index = G_F1_list.index(max(G_F1_list))
    return sum(G_F1_list)/len(G_F1_list),G_F1_list,index
    # return max(G_F1_list),index

def event_seq_match_l2_F1_single(data):
    g,G_true = data
    sum_F1 = 0
    g_event_seq_2 = []
    for i,j in g.get_edgelist():
        if g.vs['type'][i] == g.vs['type'][j]:
            continue
        g_event_seq_2.append(str(g.vs['type'][i])+'->'+str(g.vs['type'][j]))
    g_event_seq_2 = set(g_event_seq_2)
    # multiset_g_event_seq_2 = Multiset(g_event_seq_2)
    for _,g_true in enumerate(G_true):
        try:
            TP = 0
            FP = 0
            FN = 0
            g_true_event_seq_2 = []
            for i,j in g_true.get_edgelist():
                g_true_event_seq_2.append(str(g_true.vs['type'][i])+'->'+str(g_true.vs['type'][j]))
            g_true_event_seq_2 = set(g_true_event_seq_2)
            # multiset_g_true_event_seq_2 = Multiset(g_true_event_seq_2)
            # _TP = len(multiset_g_event_seq_2 & multiset_g_true_event_seq_2) 
            # _FP = len(multiset_g_event_seq_2 - multiset_g_true_event_seq_2)
            # _FN = len(multiset_g_true_event_seq_2 - multiset_g_event_seq_2)
            # for e_s in g_event_seq_2:
            #     if e_s in g_true_event_seq_2:
            #         TP+=1
            #     else:
            #         FP+=1
            # for type in g_true_event_seq_2:
            #     if type not in g_event_seq_2:
            #         FN+=1
            # # if _TP!=TP or _FP!=FP or _FN!=FN:
            # #     print()
            # pre = TP/(TP+FP)
            # rec = TP/(TP+FN)
            # F1 = 2*pre*rec/(pre+rec)
            curr_list_true = g_true_event_seq_2
            curr_list_pred = g_event_seq_2
            intersection = list((Counter(curr_list_true) & Counter(curr_list_pred)).elements())
            # if len(curr_list_pred) == 0 or len(curr_list_true) == 0:
            #     return 0., 0., 0.
            precision = len(intersection) * 1.0 / len(curr_list_pred)
            recall = len(intersection) * 1.0 / len(curr_list_true)
            if precision + recall == 0:
                f1 = 0.
            else:
                f1 = 2 * precision * recall / (precision + recall)
            F1 = f1
        except:
            F1 = 0
        sum_F1+=F1
    return sum_F1/len(G_true)

def event_seq_match_l2_F1(G,G_true):
    G_F1_list = []
    with Pool(15) as p:
        for idx,_F1 in enumerate(tqdm(p.imap(event_seq_match_l2_F1_single,zip(G,[G_true for _ in range(len(G))])),total=len(G))):
            G_F1_list.append(_F1)
    index = G_F1_list.index(max(G_F1_list))
    return sum(G_F1_list)/len(G_F1_list),G_F1_list,index

# def event_seq_match_l2_F1(G,G_true):
#     # with Pool(1) as p:
#     #     for idx,result in enumerate(tqdm(p.imap(event_seq_match_l2_F1_single,zip(guids, sources, source_tags, targets)),total=len(guids))):
#     #         examples.append(result)
#     G_F1_list = []
#     for g in G:
#         sum_F1 = 0
#         g_event_seq_2 = []
#         for i,j in g.get_edgelist():
#             g_event_seq_2.append(str(g.vs['type'][i])+'->'+str(g.vs['type'][j]))
#         for _,g_true in enumerate(tqdm(G_true,total=len(G_true))):
#             try:
#                 TP = 0
#                 FP = 0
#                 FN = 0
#                 g_true_event_seq_2 = []
#                 for i,j in g_true.get_edgelist():
#                     g_true_event_seq_2.append(str(g_true.vs['type'][i])+'->'+str(g_true.vs['type'][j]))
#                 for e_s in g_event_seq_2:
#                     if e_s in g_true_event_seq_2:
#                         TP+=1
#                     else:
#                         FP+=1
#                 for type in g_true_event_seq_2:
#                     if type not in g_event_seq_2:
#                         FN+=1
#                 pre = TP/(TP+FP)
#                 rec = TP/(TP+FN)
#                 F1 = 2*pre*rec/(pre+rec)
#             except:
#                 F1 = 0
#             sum_F1+=F1
#         G_F1_list.append(sum_F1/len(G_true))
#     index = G_F1_list.index(max(G_F1_list))
#     return sum(G_F1_list)/len(G_F1_list),G_F1_list,index

def event_seq_match_l3_F1_single(data):
    g,G_true = data
    sum_F1 = 0
    g_event_seq_3 = []
    adjlist = g.get_adjlist()
    for i in range(len(adjlist)):
        for j in adjlist[i]:
            for k in adjlist[j]:
                g_event_seq_3.append(str(g.vs['type'][i])+'->'+str(g.vs['type'][j])+'->'+str(g.vs['type'][k]))
    g_event_seq_3 = set(g_event_seq_3)
    # multiset_g_event_seq_3 = Multiset(g_event_seq_3)
    # print('l=3,num={}'.format(len(g_event_seq_3)))
    for _,g_true in enumerate(G_true):
        try:
            TP = 0
            FP = 0
            FN = 0
            g_true_event_seq_3 = []
            adjlist = g_true.get_adjlist()
            for i in range(len(adjlist)):
                for j in adjlist[i]:
                    for k in adjlist[j]:
                        g_true_event_seq_3.append(str(g.vs['type'][i])+'->'+str(g.vs['type'][j])+'->'+str(g.vs['type'][k]))
            g_true_event_seq_3 = set(g_true_event_seq_3)
            # multiset_g_true_event_seq_3 = Multiset(g_true_event_seq_3)
            # TP = len(multiset_g_event_seq_3 & multiset_g_true_event_seq_3) 
            # FP = len(multiset_g_event_seq_3 - multiset_g_true_event_seq_3)
            # FN = len(multiset_g_true_event_seq_3 - multiset_g_event_seq_3)
            # print('l=3,true_num={}'.format(len(g_true_event_seq_3)))
            # for e_s in g_event_seq_3:
            #     if e_s in g_true_event_seq_3:
            #         TP+=1
            #     else:
            #         FP+=1
            # for type in g_true_event_seq_3:
            #     if type not in g_event_seq_3:
            #         FN+=1
            # pre = TP/(TP+FP)
            # rec = TP/(TP+FN)
            # F1 = 2*pre*rec/(pre+rec)
            curr_list_true = g_true_event_seq_3
            curr_list_pred = g_event_seq_3
            intersection = list((Counter(curr_list_true) & Counter(curr_list_pred)).elements())
            # if len(curr_list_pred) == 0 or len(curr_list_true) == 0:
            #     return 0., 0., 0.
            precision = len(intersection) * 1.0 / len(curr_list_pred)
            recall = len(intersection) * 1.0 / len(curr_list_true)
            if precision + recall == 0:
                f1 = 0.
            else:
                f1 = 2 * precision * recall / (precision + recall)
            F1 = f1
        except:
            F1 = 0
        sum_F1+=F1
    return sum_F1/len(G_true)

def event_seq_match_l3_F1(G,G_true):
    G_F1_list = []
    with Pool(15) as p:
        for idx,_F1 in enumerate(tqdm(p.imap(event_seq_match_l3_F1_single,zip(G,[G_true for _ in range(len(G))])),total=len(G))):
            G_F1_list.append(_F1)
    index = G_F1_list.index(max(G_F1_list))
    return sum(G_F1_list)/len(G_F1_list),G_F1_list,index






# TP = 0
# FP = 0
# FN = 0
# A=set() # 或者是list 对于event seq match指标来说是multi-set即list
# B=set()
# for a in A:
#     if a in B:
#         TP+=1
#     else:
#         FP+=1
# for b in B:
#     if b not in A:
#         FN+=1
# pre = TP/(TP+FP)
# rec = TP/(TP+FN)
# F1 = 2*pre*rec/(pre+rec)