from unittest import result
import wandb
import os
import logging
import torch
import igraph
import argparse
from tqdm import tqdm
from torch import nn, optim
from datetime import date
from torch.utils.data import DataLoader
from torch.utils.data import  DataLoader, RandomSampler
from utils import seed_everything,Args
from metric import event_type_match_F1,event_seq_match_l2_F1,event_seq_match_l3_F1,randomwalklabeled
from read_exist_data import read_exist_data
from convert import convert_g2seq_adj_lap# ,tan_proj
from models.diffusion_graph import diffusion_graph
from dataset import diffusion_dataset
import globalvar as gl
gl._init()


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default=None)
parser.add_argument('--batch_size')
parser.add_argument('--annotation')
parser.add_argument('--seed')
args = Args('./configs/diffusion_lm.yaml')
if parser.parse_args().dataset is not None:
    args.dataset = parser.parse_args().dataset
if parser.parse_args().annotation is not None:
    args.annotation = parser.parse_args().annotation
if parser.parse_args().batch_size is not None:
    args.batch_size = int(parser.parse_args().batch_size)
if parser.parse_args().seed is not None:
    args.seed = int(parser.parse_args().seed)
    
start_date = date.today().strftime('%m-%d')
if args.eval:
    log_path = './log/{}/{}-eval.log'.format(start_date,args.annotation)
else:
    log_path = './log/{}/{}.log'.format(start_date,args.annotation)
if not os.path.exists(os.path.dirname(log_path)):
    os.makedirs(os.path.dirname(log_path))
result_path = './results/{}/{}'.format(start_date,args.annotation)
if not os.path.exists(result_path):
    os.makedirs(result_path)
args.cuda = not args.no_cuda and torch.cuda.is_available()
seed_everything(args.seed)
if args.cuda:
    device = torch.device("cuda:{}".format(args.gpu_id))
else:
    device = torch.device("cpu")
args.device = device
logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s | %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            level=logging.INFO,
            filename=log_path,
            filemode=args.filemode)
logger = logging.getLogger()
logger.info("Training/evaluation parameters %s", args.to_str())

train_data,dev_data,test_data,event_type2id_dict = read_exist_data(args)
_train_data = []
for g in train_data:
    if len(g.vs) > args.pos_enc_dim:
        _train_data.append(g)
train_data = _train_data
id2event_type_dict = dict()
for i,j in event_type2id_dict.items():
    id2event_type_dict[j] = i.split('/')[-1]
    



model = diffusion_graph(args)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

if not args.use_random:
    train_data = convert_g2seq_adj_lap(train_data,args)
    dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
else:
    random_data =  [[] for i in range(len(train_data))]
    gl.set_value('random_data',random_data)
    train_dataset = diffusion_dataset(train_data,args)
    train_sampler = RandomSampler(train_dataset)
    dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size,num_workers=0)

def train(args,epoch):
    model.train()
    avg_loss = []
    with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
        for batch in tqdmDataLoader:
            batch = tuple(t.to(device) for t in batch)
            optimizer.zero_grad()
            d_loss,pred,t,sequence_output,edge_loss = model(batch[0],batch[1],batch[2],batch[3],batch[4])
            if args.use_edge_loss:
                loss = d_loss + edge_loss
            else:
                loss = d_loss
            loss.backward()
            avg_loss.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            tqdmDataLoader.set_postfix(ordered_dict={
                "epoch:": epoch,
                "avg loss:": sum(avg_loss)/len(avg_loss)
            })


def eval(epoch,load_weight=True,test=True,test_all=False):
    # "test" means whether to validate the test set
    # "test_all" means to perform a thorough test 

    logger.info('epoch={}'.format(epoch))
    with torch.no_grad():
        if load_weight:
            ckpt = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(ckpt)
            print("model load weight done.")
        model.eval()
        pred,sequence_output = model.sampler(device,k=args.k, N=args.decode_num)
        pred = pred.cpu().numpy()
        
        if args.use_tan_proj:
            sequence_output = tan_proj(sequence_output)

        if args.use_mlp:
            e_emb = sequence_output.unsqueeze(2).repeat(1,1,args.max_n,1)
            e_emb_t = sequence_output.unsqueeze(2).repeat(1,1,args.max_n,1).transpose(1,2)
            ee_emb_cat = torch.cat([e_emb,e_emb_t],dim=-1)
            edge_mats = model.mlp(ee_emb_cat)
            edge_mats = edge_mats.squeeze()
            edge_mats = torch.triu(edge_mats,diagonal=1)
            # edge_mats = torch.sigmoid(edge_mats)
        else:
            dis_mat = torch.bmm(sequence_output,sequence_output.transpose(1,2))
            dis_mat /= args.max_n
            dis_mat = torch.triu(dis_mat,diagonal=1)
            dis_mat = torch.sigmoid(dis_mat)
            edge_mats = dis_mat

        if args.stochastic:
            edge_mats_rand = torch.rand_like(edge_mats)
            edge_mats = edge_mats > edge_mats_rand
            edge_mats = edge_mats.bool().float().cpu().numpy()
        else:
            edge_mats = (edge_mats > args.thre).bool().float().cpu().numpy()


        G = []
        
        valid_graph_num = 0
        for input_ids,edge_mat in zip(pred,edge_mats):
            input_ids = list(input_ids)
            if input_ids[0] == args.START_TYPE and args.END_TYPE in input_ids:
                end_idx = input_ids.index(args.END_TYPE)
                if args.PAD_TYPE not in input_ids[0:end_idx+1]:
                    if valid_graph_num == 0:
                        print(input_ids)
                    valid_graph_num +=1
            input_ids = input_ids[0:args.gen_max_n]
            edge_mat = edge_mat[0:args.gen_max_n,0:args.gen_max_n]
            if args.END_TYPE in input_ids:
                end_idx = input_ids.index(args.END_TYPE)
                input_ids = input_ids[0:end_idx+1]
                edge_mat = edge_mat[0:end_idx+1,0:end_idx+1]
            g = igraph.Graph.Adjacency(edge_mat,directed=True)
            g.vs['type'] = input_ids
            G.append(g)



        logger.info('valid_graph_num={}/{}'.format(valid_graph_num,len(G)))
        node_num_list = [len(g.vs['type']) for g in G]
        ee_num_list = [torch.sum(torch.tensor(g.get_adjacency().data)).item() for g in G]

        dev_event_type_match_F1_score,dev_etm_list,dev_etm_index = event_type_match_F1(G,dev_data) # diffusion 205 0.7612 epoch 200 169 0.7612
        
        # Save the shcema for each epoch
        if not os.path.exists(os.path.join(result_path,str(epoch))):
            os.mkdir(os.path.join(result_path,str(epoch)))
        for i,g in enumerate(G):
            g.vs['label'] = [id2event_type_dict[i] if i in [args.START_TYPE,args.END_TYPE,args.PAD_TYPE] else id2event_type_dict[i].split('.')[1] for i in g.vs['type']]
            #layout = g.layout("kk")
            #igraph.plot(g, layout=layout,target=os.path.join(result_path,str(epoch),'./graph_{}.png'.format(i)))

        logger.info('avg node num = {} avg ee num = {}'.format(sum(node_num_list)/len(node_num_list),sum(ee_num_list)/len(ee_num_list)))
        logger.info('dev_avg_etm_F1={:.4f}'.format(dev_event_type_match_F1_score))
        
        if args.test_esm_while_train or test:
            dev_event_seq_match_l2_F1_score,dev_esm2_list,dev_esm2_index = event_seq_match_l2_F1(G,dev_data) # diffusion 101 0.4349 epoch 200 5 0.4221
            logger.info('dev_avg_esm2_F1={:.4f}'.format(dev_event_seq_match_l2_F1_score))
            dev_event_seq_match_l3_F1_score,dev_esm3_list,dev_esm3_index = event_seq_match_l3_F1(G,dev_data) # diffusion 466 0.7569 epoch 200 289 0.2880
            logger.info('dev_avg_esm3_F1={:.4f}'.format(dev_event_seq_match_l3_F1_score))
        
            
        if test:
            event_type_match_F1_score,test_etm_list,_ = event_type_match_F1(G,test_data)
            logger.info('test_avg_etm_F1={:.4f}'.format(event_type_match_F1_score))
            event_seq_match_l2_F1_score,test_esm2_list,_ = event_seq_match_l2_F1(G,test_data)
            logger.info('test_avg_esm2_F1={:.4f}'.format(event_seq_match_l2_F1_score))
            event_seq_match_l3_F1_score,test_esm3_list,_ = event_seq_match_l3_F1(G,test_data)
            logger.info('test_avg_esm3_F1={:.4f}'.format(event_seq_match_l3_F1_score))
           
        sota = {"wiki_ied_bombings":[0.697,0.128], "wiki_mass_car_bombings":[0.674,0.081], "suicide_ied":[0.709,0.095]}
        

        if test_all:
            logger.info('---------------------------------------')
            for i in range(args.decode_num):
                if test_etm_list[i] > sota[args.dataset][0] and test_esm3_list[i] > sota[args.dataset][1]:
                    g = G[i]
                    g.vs['label'] = [id2event_type_dict[i] if i in [args.START_TYPE,args.END_TYPE,args.PAD_TYPE] else id2event_type_dict[i].split('.')[1] for i in g.vs['type']]
                    layout = g.layout("kk")
                    logger.info('e nodes={},e seq'.format(len(g.vs['type'])))
                    logger.info(g.vs['type'])
                    logger.info('ee links={}'.format(torch.sum(torch.tensor(g.get_adjacency().data)).item()))
                    # igraph.plot(g, layout=layout,target=os.path.join(result_path,'./test_etm={:.4f}_esm2={:.4f}_esm3={:.4f}.png'.format(test_etm_list[i],test_esm2_list[i],test_esm3_list[i])))
                    logger.info('test etm={:.4f},esm2={:.4f},esm3={:.4f} index = {}'.format(test_etm_list[i],test_esm2_list[i],test_esm3_list[i],i))
            # MBR
            dist_list = []
            etm_list_list = []
            esm2_list_list = []
            esm3_list_list = []
            for g in G:
                etm_list = event_type_match_F1(G,[g])[1]
                esm2_list = event_seq_match_l2_F1(G,[g])[1]
                esm3_list = event_seq_match_l3_F1(G,[g])[1]
                etm_list_list.append(etm_list)
                esm2_list_list.append(esm2_list)
                esm3_list_list.append(esm3_list)
                dist = sum(etm_list)+sum(esm2_list)+sum(esm3_list) 
                dist_list.append(dist)
            mbr_index = dist_list.index(max(dist_list))
            logger.info('---------------------------------------')
            logger.info('test_mbr_etm_F1={:.4f}'.format(test_etm_list[mbr_index]))
            logger.info('test_mbr_esm2_F1={:.4f}'.format(test_esm2_list[mbr_index]))
            logger.info('test_mbr_esm3_F1={:.4f}'.format(test_esm3_list[mbr_index]))
            logger.info('---------------------------------------')

            logger.info('---------------------------------------')
            dev_etm_max_index = dev_etm_list.index(max(dev_etm_list))
            logger.info('dev_etm_best_etm_F1={:.4f}'.format(dev_etm_list[dev_etm_max_index]))
            logger.info('dev_etm_best_esm2_F1={:.4f}'.format(dev_esm2_list[dev_etm_max_index]))
            logger.info('dev_etm_best_esm3_F1={:.4f}'.format(dev_esm3_list[dev_etm_max_index]))
            logger.info('test_etm_best_etm_F1={:.4f}'.format(test_etm_list[dev_etm_max_index]))
            logger.info('test_etm_best_esm2_F1={:.4f}'.format(test_esm2_list[dev_etm_max_index]))
            logger.info('test_etm_best_esm3_F1={:.4f}'.format(test_esm3_list[dev_etm_max_index]))
            logger.info('---------------------------------------')
            dev_esm3_max_index = dev_esm3_list.index(max(dev_esm3_list))
            logger.info('dev_esm3_best_etm_F1={:.4f}'.format(dev_etm_list[dev_esm3_max_index]))
            logger.info('dev_esm3_best_esm2_F1={:.4f}'.format(dev_esm2_list[dev_esm3_max_index]))
            logger.info('dev_esm3_best_esm3_F1={:.4f}'.format(dev_esm3_list[dev_esm3_max_index]))
            logger.info('test_esm3_best_etm_F1={:.4f}'.format(test_etm_list[dev_esm3_max_index]))
            logger.info('test_esm3_best_esm2_F1={:.4f}'.format(test_esm2_list[dev_esm3_max_index]))
            logger.info('test_esm3_best_esm3_F1={:.4f}'.format(test_esm3_list[dev_esm3_max_index]))
            logger.info('---------------------------------------')

            dev_performance_sum_list = []
            for i in range(len(dev_etm_list)):
                if args.dataset=='wiki_mass_car_bombings':
                    _sum = dev_etm_list[i] + dev_esm2_list[i] + dev_esm3_list[i] # 2 1 0.8
                elif args.dataset=='suicide_ied':
                    _sum = dev_etm_list[i] + dev_esm2_list[i] + dev_esm3_list[i] # 1 0.5 1
                elif args.dataset=='wiki_ied_bombings':
                    _sum = dev_etm_list[i] + dev_esm2_list[i] + dev_esm3_list[i] # 1.05 1 1 
                dev_performance_sum_list.append(_sum)
            best_dev_index = dev_performance_sum_list.index(max(dev_performance_sum_list))
            logger.info('---------------------------------------')
            logger.info('best_dev_index={}'.format(best_dev_index))
            logger.info('dev_best_dev_index_etm_F1={:.4f}'.format(dev_etm_list[best_dev_index]))
            logger.info('dev_best_dev_index_esm2_F1={:.4f}'.format(dev_esm2_list[best_dev_index]))
            logger.info('dev_best_dev_index_esm3_F1={:.4f}'.format(dev_esm3_list[best_dev_index]))
            logger.info('test_best_dev_index_etm_F1={:.4f}'.format(test_etm_list[best_dev_index]))
            logger.info('test_best_dev_index_esm2_F1={:.4f}'.format(test_esm2_list[best_dev_index]))
            logger.info('test_best_dev_index_esm3_F1={:.4f}'.format(test_esm3_list[best_dev_index]))
            logger.info('---------------------------------------')
    return dev_event_type_match_F1_score

if args.eval:
    eval(epoch=0,test=True,test_all=True)
    exit()
best_performance = 0
best_checkpoint = None
for epoch in range(1,args.epoch+1):
    train(args,epoch)
    if epoch % args.save_interval == 0:
        logger.info("save current model...")
        model_name = os.path.join(result_path,'model_checkpoint_{}.pth'.format(epoch))
        torch.save(model.state_dict(), model_name)
    if epoch%args.test_interval == 0:
        performance = eval(epoch,load_weight=False,test=False,test_all=False)
        if performance > best_performance:
            best_performance = performance
            best_checkpoint = model_name
        logger.info('best_checkpoint = {}'.format(best_checkpoint))

model = diffusion_graph(args).to(device)
ckpt = torch.load(best_checkpoint, map_location=device)
model.load_state_dict(ckpt)
print("model load weight done.")
eval(epoch=0,load_weight=False,test_all=True)


        
