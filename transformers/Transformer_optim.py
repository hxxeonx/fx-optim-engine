import torch
from torch import nn

from .layers.Embed import DataEmbedding
from .layers.EncDec import Base_Encoder, IEBlock
from .layers.SelfAttention import Attention

class Basic_Transformer(nn.Module):
    def __init__(self, cfg):
        super(Basic_Transformer, self).__init__()
        
        self.lookback       = int(cfg.dataset.time_interval)    # lookback
        self.lookahead      = int(cfg.dataset.f_horizon)        # lookahead
        self.num_node       = int(cfg.model.enc_in)             # Feature 수 (close 1개)
        self.num_classes    = cfg.model.num_classes             # label class 수 

        self.d_model        = cfg.model.d_model                 # (학습 데이터 >> 특징 벡터) 변환 차원 
        self.d_ff           = cfg.model.d_ff                    # feed-forward 차원 
        self.c_dim          = None
        self.n_heads        = cfg.model.n_heads                 # Attention Head 수 
        
        self.pe_method      = cfg.model.pe_method               # 위치 정보 벡터 생성 알고리즘 선택
        self.learn_pe       = cfg.model.learn_pe                # Positional layer 학습 여부 (?)

        self.norm           = cfg.model.norm                    # norm 기법 선택 
        self.pre_norm       = cfg.model.pre_norm                # pre-norm 여부 
        self.activation     = cfg.model.activation              # activation function 선택 
        self.dropout        = cfg.model.dropout                 # dropout rate
        self.attn_dropout   = cfg.model.attn_dropout            # dropout rate (in Attention)

        self.res_attention  = cfg.model.res_attention           # residual attention 선택 여부 
        self.n_layers       = cfg.model.e_layers                # Encoder/Decoder layer 수 
        self.output_attention = cfg.model.output_attention      # attention value output 여부 

        self.cfg = cfg
        self._build()
        
    def _build(self):
        
        self.enc_embedding  = DataEmbedding( t_interval = self.lookback,
                                             n_vars     = self.num_node,
                                             d_model    = self.d_model, 
                                             pe_method  = self.pe_method, 
                                             learn_pe   = self.learn_pe, 
                                             dropout    = 0.1)

        self.encoder        = Base_Encoder( Attention, 
                                            self.d_model, 
                                            self.n_heads, 
                                            d_ff = self.d_ff,
                                            norm = self.norm, 
                                            attn_dropout     = self.attn_dropout,
                                            dropout          = self.dropout,
                                            pre_norm         = self.pre_norm, 
                                            activation       = self.activation, 
                                            res_attention    = self.res_attention, 
                                            n_layers         = self.n_layers, 
                                            output_attention = self.output_attention)

        self.layer_3        = IEBlock( input_dim  = self.d_model * 2,
                                       hid_dim    = self.d_model,
                                       output_dim = self.lookahead,
                                       num_node   = self.num_node,
                                       c_dim      = self.c_dim)

        self.ar             = nn.Linear(self.lookback, self.lookahead)

        self.mlp_head = nn.Sequential(
            nn.BatchNorm1d(self.lookahead * self.num_node),
            nn.Linear(self.lookahead * self.num_node, self.num_classes)
            # nn.Dropout(cfg.model.dropout),
            # nn.ReLU(), 
            # nn.Linear(int(cfg.model.d_model/2), self.num_classes)
        )

    def forward(self, x):

        # Batch-Normalization
        # mean_enc = x[:, :, :].mean(1, keepdim = True).detach() # B x 1 x E
        # std_enc  = torch.sqrt(torch.var(x[:, :, :], dim = 1, keepdim = True, unbiased = False) + 1e-5).detach() # B x 1 x E

        # x[:, :, :] = x[:, :, :] - torch.mean(mean_enc[: , : , :], dim = 2, keepdim = True)
        # x[:, :, :] = x[:, :, :] / torch.mean(std_enc[: , : , :], dim = 2, keepdim=True)


        try:
            B, T, N = x.size()
            highway = self.ar(x.permute(0, 2, 1))                                   
            highway = highway.permute(0, 2, 1)
                                                                                    # x: [bs x nvars x time_inteval]
            x      = x.permute(0, 1, 2)                                             # x: [bs x time_inteval x nvars]
            x      = self.enc_embedding(x)                                          # x: [bs * nvars x d_model] (특징값 + 위치 정보)
            x      = self.encoder(x)                                                # x: [bs * nvars x d_model] ()
            x      = x.permute(0,2,1)                                               # x: [bs * nvars x d_model x patch_num]

            out = self.layer_3(x)
            out = out + highway
            out = torch.flatten(out,start_dim=1)
            out = self.mlp_head(out)

            if self.smax:
                return torch.softmax(out, dim=1)
            else:
                return out
            
        except Exception as err:
            import pdb; pdb.set_trace()
