import math
import torch
import torch.nn as nn
import torch.nn.functional as F 
from transformers import AutoTokenizer
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.models.bert.configuration_bert import BertConfig
from .modeling import MyBertForMaskedLM
# from convert import tan_proj

class diffusion_graph(nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        bert_config = BertConfig.from_pretrained(args.model_config)
        self.base_model = MyBertForMaskedLM(bert_config)
        self.edge_model = MyBertForMaskedLM(bert_config)
        self.node_model = MyBertForMaskedLM(bert_config)
        self.max_len = args.max_n
        self.max_step = args.diff_step
        self.time_embed = nn.Embedding(self.max_step,self.base_model.config.hidden_size)
        self.embedding_lap_pos_enc = nn.Linear(args.pos_enc_dim,self.base_model.config.hidden_size)
        self.args = args
        nn.init.constant_(self.time_embed.weight, 0)
        self.mlp = nn.Sequential(
                nn.Linear(self.base_model.config.hidden_size*2, self.base_model.config.hidden_size*4), 
                nn.ReLU(), 
                nn.Linear(self.base_model.config.hidden_size*4, 1),
                nn.Sigmoid()
                )

    def forward(self,input_ids,attention_mask,edge_matrix,lap_pos_enc,weight_tensor,t=None):
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        
        position_ids = self.base_model.bert.embeddings.position_ids[:, 0 : seq_length]
        position_embeddings = self.base_model.bert.embeddings.position_embeddings(position_ids)
        h_lap_pos_enc = self.embedding_lap_pos_enc(lap_pos_enc) 
        word_emb = self.base_model.bert.embeddings.word_embeddings(input_ids)
        if self.args.use_lap:
            word_emb = word_emb + h_lap_pos_enc
        if t is None:
            diffusion_steps = torch.randint(0,self.max_step,size = (input_shape[0],),device=input_ids.device)
        else:
            diffusion_steps = torch.ones(size = (input_shape[0],),device=input_ids.device).long()*t



        noise = torch.randn_like(word_emb)/math.sqrt(self.base_model.config.hidden_size)
        alpha = 1 - torch.sqrt((diffusion_steps+1)/self.max_step).view(-1,1,1)
        noisy_word = torch.sqrt(alpha)*word_emb+torch.sqrt(1-alpha)*noise#  + token_type_embeddings


            
        time_embedding = self.time_embed(diffusion_steps).unsqueeze(1)
        noisy_word = noisy_word+position_embeddings+time_embedding

        if self.args.lm_loss:
            zero_diffusion_steps = torch.ones(size = (input_shape[0],),device=input_ids.device).long()*0
            alpha_0 = 1 - torch.sqrt((zero_diffusion_steps+1)/self.max_step).view(-1,1,1)
            alpha_1 = 1 - torch.sqrt((zero_diffusion_steps+1)/self.max_step).view(-1,1,1)
            x_0 = torch.sqrt(alpha_0)*word_emb+torch.sqrt(1-alpha_0)*noise
            x_1 = torch.sqrt(alpha_1)*word_emb+torch.sqrt(1-alpha_1)*noise
            x_1_pos_time = x_1+position_embeddings+time_embedding
        middle_noisy_word = self.base_model.bert.embeddings.LayerNorm(noisy_word)

        extended_attention_mask = self.base_model.bert.get_extended_attention_mask(attention_mask, input_shape,device=attention_mask.device)
        
        middle_encoder_outputs = self.base_model.bert.encoder(
            middle_noisy_word,
            attention_mask=extended_attention_mask
        )
        middle_sequence_output = middle_encoder_outputs[0]
        node_encoder_outputs = self.node_model.bert.encoder(
            middle_sequence_output,
            attention_mask=extended_attention_mask
        )
        edge_encoder_inputs = middle_sequence_output + position_embeddings
        edge_encoder_outputs = self.edge_model.bert.encoder(
            edge_encoder_inputs,
            attention_mask=extended_attention_mask
        )
        node_encoder_outputs = node_encoder_outputs[0] 

        if self.args.lm_loss:
            loss_1 = torch.mean(torch.sum(torch.pow((node_encoder_outputs-x_0),2),-1))
            p_x_1 = self.base_model.bert.embeddings.LayerNorm(x_1_pos_time)
            p_x_1 = self.base_model.bert.encoder(p_x_1,attention_mask=extended_attention_mask)[0]
            p_x_1 = self.node_model.bert.encoder(p_x_1,attention_mask=extended_attention_mask)[0]
            loss_2 = torch.mean(torch.sum(torch.pow((word_emb-p_x_1),2),-1))
            prediction_scores = self.node_model.cls.predictions(x_0)
            loss_3 = F.cross_entropy(prediction_scores.view(-1, self.node_model.config.vocab_size),input_ids.flatten(),ignore_index=self.args.PAD_TYPE,reduction='mean')
            loss = loss_1+loss_2+loss_3
        else:
            prediction_scores = self.node_model.cls.predictions(node_encoder_outputs)
            loss = F.cross_entropy(prediction_scores.view(-1, self.node_model.config.vocab_size),input_ids.flatten(),ignore_index=self.args.PAD_TYPE,reduction='mean')
        
        edge_encoder_outputs = edge_encoder_outputs[0]
    
        if self.args.use_tan_proj:
            edge_encoder_outputs = tan_proj(edge_encoder_outputs)
        if self.args.use_mlp:
            e_emb = edge_encoder_outputs.unsqueeze(2).repeat(1,1,self.args.max_n,1)
            e_emb_t = edge_encoder_outputs.unsqueeze(2).repeat(1,1,self.args.max_n,1).transpose(1,2)
            ee_emb_cat = torch.cat([e_emb,e_emb_t],dim=-1)
            ee_p = self.mlp(ee_emb_cat)
            ee_p = ee_p.squeeze()
            dis_mat = torch.triu(ee_p,diagonal=1)
        else:
            dis_mat = torch.bmm(edge_encoder_outputs,edge_encoder_outputs.transpose(1,2))
            dis_mat = torch.sigmoid(dis_mat)
            dis_mat = torch.triu(dis_mat,diagonal=1)
        edge_loss_fun = torch.nn.MSELoss()
        edge_loss = edge_loss_fun(dis_mat,edge_matrix)
        # edge_loss = F.binary_cross_entropy(dis_mat.view(-1), edge_matrix.view(-1), weight = weight_tensor.view(-1))


        return loss,prediction_scores,diffusion_steps,edge_encoder_outputs,edge_loss

    @torch.no_grad()
    def sampler(self,device,k=1,N=100):
        import time
        
        start_time = time.time()
        noisy_word = torch.normal(0,1,(N,self.max_len,self.base_model.config.hidden_size)).to(device) / math.sqrt(self.base_model.config.hidden_size)
        attention_mask = torch.ones(N,self.max_len).long().to(device)
        extended_attention_mask = self.base_model.bert.get_extended_attention_mask(attention_mask, attention_mask.shape,device=device)

        position_ids = self.base_model.bert.embeddings.position_ids[:, 0 : self.max_len]
        position_embeddings = self.base_model.bert.embeddings.position_embeddings(position_ids)

        for t in range(self.max_step-1,0,-k):
            diffusion_steps = torch.ones(size = (N,),device=device).long()*t
            time_embedding = self.time_embed(diffusion_steps).unsqueeze(1)

            model_input = noisy_word +position_embeddings+time_embedding # +token_type_embeddings
            model_input = self.base_model.bert.embeddings.LayerNorm(model_input)
            middle_encoder_outputs = self.base_model.bert.encoder(
                model_input,
                attention_mask=extended_attention_mask,
                head_mask=[None] * self.base_model.config.num_hidden_layers
            )
            middle_sequence_output = middle_encoder_outputs[0]
            node_encoder_outputs = self.node_model.bert.encoder(
                middle_sequence_output,
                attention_mask=extended_attention_mask
            )
            edge_encoder_inputs = middle_sequence_output + position_embeddings
            edge_encoder_outputs = self.edge_model.bert.encoder(
                edge_encoder_inputs,
                attention_mask=extended_attention_mask
            )
            edge_encoder_output = edge_encoder_outputs[0]
            node_encoder_output = node_encoder_outputs[0]
            prediction_scores = self.node_model.cls.predictions(node_encoder_output)

            pred = torch.argmax(prediction_scores,-1).long()
            denoised_word = self.base_model.bert.embeddings.word_embeddings(pred)
            # denoised_word = prediction_scores.softmax(-1) @ self.model.bert.embeddings.word_embeddings.weight.unsqueeze(0)
        
            alpha_tk = 1 - math.sqrt((t+1-k)/self.max_step)#+1e-5
            alpha_t = 1 - math.sqrt((t+1)/self.max_step)+1e-5

            noise = (noisy_word - math.sqrt(alpha_t)*denoised_word)/math.sqrt(1-alpha_t)
            if self.args.edge_loop:
                noisy_word = math.sqrt(alpha_tk)*edge_encoder_output + math.sqrt(1-alpha_tk)*noise
            else:
                # noisy_word = math.sqrt(alpha_tk)*(noisy_word/math.sqrt(alpha_t) + (math.sqrt((1-alpha_tk)/alpha_tk) - math.sqrt((1-alpha_t)/alpha_t))*noise)
                noisy_word = math.sqrt(alpha_tk)*denoised_word + math.sqrt(1-alpha_tk)*noise
            print(f"\rnoise level {t}  {time.time()-start_time:.2f}",end='')
        if self.args.node_stochastic:
            pred = torch.multinomial(prediction_scores.softmax(-1).reshape(-1,self.args.num_vertex_type),1).reshape(self.args.decode_num,self.args.max_n)
        else:
            pred = torch.argmax(prediction_scores,-1).long()

        return pred,edge_encoder_output