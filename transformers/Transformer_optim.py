import torch
from torch import nn

from .layers.Embed import DataEmbedding
from .layers.EncDec import Base_Encoder
from .layers.SelfAttention import Attention

class Basic_Transformer(nn.Module):
    def __init__(self, cfg):
        super(Basic_Transformer, self).__init__()
        
        self.lookback       = int(cfg.dataset.time_interval)
        self.num_node       = int(cfg.model.enc_in)             # Feature 수 (close 1개)

        self.d_model        = cfg.model.d_model                 # (학습 데이터 >> 특징 벡터) 변환 차원 
        self.pe_method      = cfg.model.pe_method               # 위치 정보 벡터 생성 알고리즘 
        self.learn_pe       = cfg.model.learn_pe

        self.cfg = cfg
        self._build()
        
    def _build(self):
        
        self.enc_embedding = DataEmbedding( t_interval = self.lookback,
                                            n_vars     = self.num_node,
                                            d_model    = self.d_model, 
                                            pe_method  = self.pe_method, 
                                            learn_pe   = self.learn_pe, 
                                            dropout    = 0.1)

        # self.encoder = Base_Encoder(Attention, 
        #                             self.num_chunks, 
        #                             self.d_model, 
        #                             self.n_heads, 
        #                             d_k  = self.d_k, 
        #                             d_v  = self.d_v, 
        #                             d_ff = self.d_ff,
        #                             norm = self.norm, 
        #                             attn_dropout     = self.attn_dropout,
        #                             dropout          = self.dropout,
        #                             pre_norm         = self.pre_norm, 
        #                             activation       = self.activation, 
        #                             res_attention    = self.res_attention, 
        #                             n_layers         = self.n_layers, 
        #                             output_attention = self.output_attention)
        

        # self.chunk_proj_1 = nn.Linear(self.num_chunks, 1)

        # self.layer_3 = IEBlock(
        #     input_dim  = self.d_model * 2,
        #     hid_dim    = self.d_model,
        #     output_dim = self.lookahead,
        #     num_node   = self.num_node,
        #     c_dim      = self.c_dim
        # )

        # self.ar = nn.Linear(self.lookback, self.lookahead)

        # self.mlp_head = nn.Sequential(
        #     nn.BatchNorm1d(self.lookahead * self.num_node),
        #     nn.Linear(self.lookahead * self.num_node, self.num_classes)
        #     # nn.Dropout(cfg.model.dropout),
        #     # nn.ReLU(), 
        #     # nn.Linear(int(cfg.model.d_model/2), self.num_classes)
        # )

    def forward(self, x):

        ## (B,1,time_interval,4)
        # x = x.squeeze(1)
        ## (B,time_interval,4)
        ## Batch size / time interval / OHLC

        # Batch-Normalization
        # mean_enc = x[:, :, :].mean(1, keepdim = True).detach() # B x 1 x E
        # std_enc  = torch.sqrt(torch.var(x[:, :, :], dim = 1, keepdim = True, unbiased = False) + 1e-5).detach() # B x 1 x E

        # x[:, :, :] = x[:, :, :] - torch.mean(mean_enc[: , : , :], dim = 2, keepdim = True)
        # x[:, :, :] = x[:, :, :] / torch.mean(std_enc[: , : , :], dim = 2, keepdim=True)

        
        # try:

            # x       = self.W_emb(x.long())
        try:
            B, T, N = x.size()
            # highway = self.ar(x.permute(0, 2, 1))
            # highway = highway.permute(0, 2, 1)

            # x: [bs x nvars xtime_inteval]
            x      = x.permute(0, 1, 2)                                              # x: [bs x time_inteval x nvars]
            x      = self.enc_embedding(x)                                           # z: [bs * nvars x d_model]
            import pdb; pdb.set_trace()

            x      = self.encoder(x)                                                 # z: [bs * nvars x d_model]
            x      = x.permute(0,2,1)                                                  # z: [bs * nvars x d_model x patch_num]
            x      = self.chunk_proj(x).squeeze(dim=-1)                              # z: [bs * nvars x d_model x 1]

            out = self.layer_3(x)

            out = out # + highway
            
            out = torch.flatten(out,start_dim=1)
            out = self.mlp_head(out)

            if self.smax:
                return torch.softmax(out, dim=1)
            else:
                return out
            
        except Exception as err:
            print(err)
            print(x.type())
            
            import pdb; pdb.set_trace()
