from math import sqrt

import numpy as np
import torch
import torch.nn as nn
from dgllife.model.gnn.gat import GAT
from dgl.nn.pytorch import Set2Set
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import tokens_struct
from torch_geometric.nn import GraphNorm
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize

from torch_geometric.nn.aggr import AttentionalAggregation, SortAggregation
from torch.nn.parameter import Parameter
from torch import Tensor
from torch.nn import init
import math



from transformers import AutoTokenizer, AutoModelForMaskedLM
path='/root/autodl-tmp/MultiCBlo-master/seyonec450k/PubChem10M_SMILES_BPE_450k'
tokenizer = AutoTokenizer.from_pretrained(path)
model_450 = AutoModelForMaskedLM.from_pretrained(path)

class SMILESFeatureExtractor(nn.Module):
    def __init__(self, model, tokenizer, output_dim=384, device='cpu'):
        super(SMILESFeatureExtractor, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.linear = nn.Linear(model.config.vocab_size, output_dim)  
        self.to(device)   

    def forward(self, smiles_list):
        encoded = self.tokenizer(
            smiles_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        input_ids = encoded['input_ids'].to(self.model.device)
        attention_mask = encoded['attention_mask'].to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            if hasattr(outputs, 'last_hidden_state'):
                cls_token_embedding = outputs.last_hidden_state[:, 0, :]   
            else:
                cls_token_embedding = outputs[0][:, 0, :]   
            transformed_output = self.linear(cls_token_embedding)    
        
        return transformed_output

class AttentionalAggregation(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, batch=None, dim_size=None):
        attention_weights = torch.softmax(x, dim=-1)
        aggregated_output = x * attention_weights  
        return aggregated_output

class GlobalAttention(AttentionalAggregation):
    def __call__(self, x, batch=None, size=None):
        return super().__call__(x, batch, dim_size=size)
    
class Global_Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.at = GlobalAttention()
    def forward(self, x, batch):
        return self.at(x, batch)


class WeightFusion(nn.Module):

    def __init__(self, feat_views, feat_dim, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(WeightFusion, self).__init__()
        self.feat_views = feat_views
        self.feat_dim = feat_dim
        self.weight = Parameter(torch.empty((1, 1, feat_views), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(int(feat_dim), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:

        return sum([input[i]*weight for i, weight in enumerate(self.weight[0][0])]) + self.bias

loss_type = {'reg': nn.MSELoss(reduction="none")}

class Model(nn.Module):
    def __init__(self, in_feats=64, hidden_feats=None, num_step_set2set=6,
                 num_layer_set2set=3, rnn_embed_dim=64, blstm_dim=128, blstm_layers=2, fp_2_dim=128, num_heads=4,
                 dropout=0.2, device='cpu'):
        super(Model, self).__init__()
        self.device = device
        self.vocab = tokens_struct()
        if hidden_feats is None:
            hidden_feats = [64, 64]
        self.final_hidden_feats = hidden_feats[-1]
        self.gnn = GNNModule(in_feats, hidden_feats, dropout, num_step_set2set, num_layer_set2set)
        # self.rnn = RNNModule(self.vocab, rnn_embed_dim, blstm_dim, blstm_layers, self.final_hidden_feats, dropout,
        #                      bidirectional=True, device=device)

        self.feature_extractor = SMILESFeatureExtractor(model_450, tokenizer, output_dim=384, device=device)
        

        # self.model = EGNN(
        #     in_node_nf=in_feats,
        #     hidden_nf=64,
        #     out_node_nf=64,  
        #     in_edge_nf=1, 
        #     device=device,
        #     act_fn=nn.SiLU(),
        #     n_layers=4, 
        #     residual=True,
        #     attention=False,
        #     normalize=False,
        #     tanh=False
        # )
        
        self.fp_mlp = FPNModule(fp_2_dim, self.final_hidden_feats)
        self.conv = nn.Sequential(nn.Conv2d(12, 12, kernel_size=3), nn.ReLU(),
                                  nn.Dropout(dropout))

        self.mlp1 = nn.Sequential(
            nn.Linear(384, 256),

            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
        self.pool = Global_Attention(384).to(self.device)


        self.fusion = WeightFusion(num_layer_set2set, 384, device=self.device)
        self.Layernorm = nn.Sequential(
            nn.Linear(384, 384),   
            nn.ReLU(inplace=True),                 
            nn.LayerNorm(384),              
            nn.Linear(384, 384)      
        )
        self.sigmoid = nn.Sigmoid()
        self.entropy = loss_type['reg']


    def forward(self, padded_smiles_batch, batch, fp_t):
        
        ck=list()
        batch_size = padded_smiles_batch.size(0)

        # smiles_rnn=self.rnn(padded_smiles_batch, batch.seq_len)

        smiles_features = self.feature_extractor(batch.smiles_str) 
        smiles_pool=self.pool(smiles_features,batch_size)
        # smiles_pool=smiles_pool.zero_()
        # ck.append(smiles_pool)
        ck.append(self.Layernorm(smiles_pool))
        


        
        fp_x=self.pool(self.fp_mlp(fp_t),batch_size)
        # fp_x=fp_x.zero_()
        # ck.append(fp_x)
        ck.append(self.Layernorm(fp_x))



        graph, loss2 = self.gnn(batch.x, batch.edge_index, batch.batch)
        graph_x = self.pool(graph,batch_size)
        # graph_x=graph_x.zero_()
        # ck.append(graph_x)
        ck.append(self.Layernorm(graph_x))




        # h, x = self.model(batch.x, batch.coords, batch.edge_index, batch.edge_attr)
        # graph_3D=aggregate_graph_features(h,x,batch_size)
        # graph_3D_x = self.pool(graph_3D,batch_size)
        # ck.append(self.Layernorm(graph_3D_x))d    


        molecule_emb = self.fusion(torch.stack(ck, dim=0))
        out=molecule_emb.view(batch_size, 1, -1)
        

        out = self.mlp1(out.squeeze(1))

        return out, ck #ck = ck and ck=loss

    def predict(self, smiles, graphs, atom_feats, fp_t):
        return self.sigmoid(self.forward(smiles, graphs, atom_feats, fp_t))

    
    # def label_loss(self, pred, label, mask):
    #     loss_mat = self.entropy(pred, label)
    #     return loss_mat.sum() / mask.sum()
    def label_loss(self, pred, label):
        loss_mat = self.entropy(pred, label)
        return loss_mat.mean() 

    def cl_loss(self, x1, x2, T=0.1):
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss1).mean()
        return loss


    def loss_cal(self, x_list, pred, label, alpha=0.08):
        loss1 = self.label_loss(pred, label)
        loss2 = torch.tensor(0, dtype=torch.float).to(self.device)
        modal_num = len(x_list)
        for i in range(modal_num):
            loss2 += self.cl_loss(x_list[i], x_list[i-1])

        return loss1 + alpha * loss2, loss1, loss2

class GNNModule(nn.Module):
    def __init__(self, in_feats=64, hidden_feats=None, dropout=0.2, num_step_set2set=6,
                 num_layer_set2set=3):
        super(GNNModule, self).__init__()
        self.conv = GAT(in_feats, hidden_feats)
        self.readout = Set2Set(input_dim=hidden_feats[-1],
                               n_iters=num_step_set2set,
                               n_layers=num_layer_set2set)
        self.norm = GraphNorm(hidden_feats[-1] * 2)
        self.fc = nn.Sequential(nn.Linear(hidden_feats[-1] * 2, hidden_feats[-1]), nn.ReLU(),
                                nn.Dropout(p=dropout))
        num_features_xd = 84

        self.conv1 = GINConv(nn.Linear(num_features_xd, num_features_xd))
        self.conv2 = GINConv(nn.Linear(num_features_xd, num_features_xd * 10))
        self.fc_g = nn.Sequential(
            nn.Linear(num_features_xd * 10 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 384)
        )
        self.relu = nn.ReLU()
        self.vq = VectorQuantize(dim=num_features_xd * 10,
                                 codebook_size=4000,
                                 commitment_weight=0.1,
                                 decay=0.9)

    def vector_quantize(self, f, vq_model):
        v_f, indices, v_loss = vq_model(f)

        return v_f, v_loss

    def forward(self, x, edge_index, batch):
        x_g = self.relu(self.conv1(x, edge_index))
        x_g = self.relu(self.conv2(x_g, edge_index))
        # node_v_feat, cmt_loss = self.vector_quantize(x_g.unsqueeze(0), self.vq)
        # node_v_feat = node_v_feat.squeeze(0)
        # node_res_feat = x_g + node_v_feat
        x_g = torch.cat([gmp(x_g, batch), gap(x_g, batch)], dim=1)
        x_g = self.fc_g(x_g)
        return x_g, 0


class RNNModule(nn.Module):
    def __init__(self, vocab, embed_dim, blstm_dim, num_layers, out_dim=2, dropout=0.2, bidirectional=True,
                 device='cpu'):
        super(RNNModule, self).__init__()
        self.W_rnn = nn.GRU(bidirectional=True, num_layers=1, input_size=100, hidden_size=100)
        self.fc = nn.Sequential(
            nn.Linear(200, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        self.linear = nn.Sequential(
            nn.Linear(200, 512),
            nn.Linear(512, 384)
        )

    def forward(self, batch, seq_len):
        smi_em = batch.view(-1, 100, 100).float()
        smi_em, _ = self.W_rnn(smi_em)
        smi_em = torch.relu(smi_em)
        sentence_att = self.softmax(torch.tanh(self.fc(smi_em)), 1)
        smi_em = torch.sum(sentence_att.transpose(1, 2) @ smi_em, 1) / 10
        smi_em = self.linear(smi_em)
        return smi_em

    @staticmethod
    def softmax(input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        soft_max_2d = F.softmax(trans_input.contiguous().view(-1, trans_input.size()[-1]), dim=1)
        return soft_max_2d.view(*trans_input.size()).transpose(axis, len(input_size) - 1)


class FPNModule(nn.Module):
    def __init__(self, fp_2_dim, out_feats, dropout=0.2):
        super(FPNModule, self).__init__()
        self.fp_2_dim = fp_2_dim
        self.dropout_fpn = dropout
        self.out_feats = out_feats
        self.fp_dim = 2513
        self.fc1 = nn.Linear(self.fp_dim, self.fp_2_dim)
        self.act_func = nn.ReLU()
        self.fc2 = nn.Linear(self.fp_2_dim, self.out_feats)
        self.dropout = nn.Dropout(p=self.dropout_fpn)

    def forward(self, smiles):
        fpn_out = self.fc1(smiles)
        fpn_out = self.dropout(fpn_out)
        fpn_out = self.act_func(fpn_out)
        fpn_out = self.fc2(fpn_out)
        return fpn_out


class Separator(nn.Module):
    def __init__(self):
        super(Separator, self).__init__()
        # if args.dataset.startswith('GOOD'):
        #     # GOOD
        #     # if config.model.model_name == 'GIN':
        #     #     self.r_gnn = GINFeatExtractor(config, without_readout=True)
        #     # else:
        #     #     self.r_gnn = vGINFeatExtractor(config, without_readout=True)
        #     emb_d = config.model.dim_hidden
        # else:
        #     self.r_gnn = GNN_node(num_layer=args.layer, emb_dim=args.emb_dim,
        #                           drop_ratio=args.dropout, gnn_type=args.gnn_type)
        # num_features_xd=93

        emb_d = 1152

        self.separator = nn.Sequential(nn.Linear(emb_d, emb_d * 2),
                                       # nn.BatchNorm1d(emb_d * 2),
                                       nn.ReLU(),
                                       nn.Linear(emb_d * 2, emb_d),
                                       nn.Sigmoid())
        # self.args = args
        # self.conv1 = GINConv(nn.Linear(num_features_xd, num_features_xd))
        # self.conv2 = GINConv(nn.Linear(num_features_xd, num_features_xd * 10))
        self.relu = nn.ReLU()

    def forward(self, data):
        # x_g = self.relu(self.conv1(data.x, data.edge_index))
        #
        # x_g = self.relu(self.conv2(x_g, data.edge_index))
        score = self.separator(data)  # [n, d]

        # reg on score

        pos_score_on_node = score.mean(1)  # [n]
        pos_score_on_batch = pos_score_on_node  # [B]
        neg_score_on_batch = 1 - pos_score_on_node  # [B]
        return score, pos_score_on_batch + 1e-8, neg_score_on_batch + 1e-8
