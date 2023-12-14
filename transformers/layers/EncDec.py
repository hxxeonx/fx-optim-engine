from typing import Optional

import torch
from torch import nn
from torch import Tensor

def get_activation_fn(activation):
    if callable(activation): return activation()
    elif activation.lower() == "relu": return nn.ReLU()
    elif activation.lower() == "gelu": return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable') 

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super(Transpose,self).__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

class IEBlock(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, num_node, c_dim=None):
        super(IEBlock, self).__init__()

        self.num_node   = num_node
        self.input_dim  = input_dim
        self.hid_dim    = hid_dim
        self.output_dim = output_dim

        if c_dim is None:
            self.c_dim = self.num_node // 2
        else:
            self.c_dim = c_dim

        self._build()

    def _build(self):
        self.spatial_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hid_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hid_dim, self.hid_dim // 4)
        )

        self.channel_proj = nn.Linear(self.num_node, self.num_node)
        torch.nn.init.eye_(self.channel_proj.weight)

        self.output_proj = nn.Linear(self.hid_dim // 4, self.output_dim)


    def forward(self, x):
        x = self.spatial_proj(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1) + self.channel_proj(x.permute(0, 2, 1))
        x = self.output_proj(x.permute(0, 2, 1))

        x = x.permute(0, 2, 1)

        return x        

class Base_Encoder(nn.Module):
    def __init__(self, 
                 MultiheadAttention, 
                 d_model,                              
                 n_heads, 
                 d_k           = None, 
                 d_v           = None, 
                 d_ff          = None, 
                 norm          = 'BatchNorm' , 
                 attn_dropout  = 0. , 
                 dropout       = 0. , 
                 activation    = 'gelu' ,
                 res_attention = False, 
                 n_layers      = 1,
                 pre_norm      = False,
                 output_attention = False):
        

        super(Base_Encoder,self).__init__()

        ## Module을 리스트 형태로 저장하여 정의된 모듈들에 인덱스로 접근이 가능하다. 
        self.layers = nn.ModuleList([EncoderLayer(MultiheadAttention, 
                                                  d_model, 
                                                  n_heads = n_heads, 
                                                  d_k = d_k, d_v = d_v, d_ff = d_ff, 
                                                  norm = norm,
                                                  attn_dropout = attn_dropout, 
                                                  dropout = dropout,
                                                  activation = activation, 
                                                  res_attention  =  res_attention,
                                                  pre_norm = pre_norm, 
                                                  output_attention = output_attention) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, 
                src:Tensor, 
                key_padding_mask:Optional[Tensor] = None, 
                attn_mask:Optional[Tensor] = None):
        
        output = src
        scores = None

        if self.res_attention:
            for mod in self.layers: 
                output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
                import pdb; pdb.set_trace()
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output

class EncoderLayer(nn.Module):
    def __init__(self, MultiheadAttention, d_model, n_heads, 
                 d_k = None, d_v = None, d_ff = 256, 
                 output_attention = False,
                 norm = 'BatchNorm', 
                 attn_dropout = 0, 
                 dropout = 0., 
                 bias = True, 
                 activation = "gelu", 
                 res_attention = False, 
                 pre_norm = False):
        super(EncoderLayer,self).__init__()

        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn     = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.output_attention = output_attention


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:
        
        # Pre-LN
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)


        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, 
                                                key_padding_mask=key_padding_mask, 
                                                attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, 
                                        key_padding_mask=key_padding_mask, 
                                        attn_mask=attn_mask)

        if self.output_attention:
            self.attn = attn

        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)

        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        import pdb; pdb.set_trace()
        if self.res_attention:
            return src, scores
        else:
            return src