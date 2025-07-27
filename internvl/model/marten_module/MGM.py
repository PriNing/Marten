from typing import List, Optional, Tuple, Union
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlockDecoder(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, q, k, v, im_m):
        self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=q.device) if self.attn_mask is not None else None
        return self.attn(q, k, v, attn_mask=self.attn_mask, key_padding_mask=im_m)

    def forward(self, x: list):
        if len(x) == 4:
            q, k, v, im_m = x
        else:
            q, k, v, im_m, m = x
        
        q_, m = self.attention(q, k, v, im_m)
        q = q + self.ln_1(q_)
        q = q + self.mlp(self.ln_2(q))
        return [q, k, v, im_m, m]


class TransformerDecoder(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlockDecoder(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x):
        return self.resblocks(x)


class MGM(nn.Module):
    def __init__(self, input_size, hidden_size, dev_convs_nums, out_channels, layer_num=2, head_num=8):
        
        super(MGM, self).__init__()
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.head_num = head_num
        self.dev_convs_nums = dev_convs_nums

        self.proj_q = nn.Linear(input_size, hidden_size)
        self.proj_kv = nn.Linear(input_size, hidden_size)

        self.ln_q= nn.LayerNorm(hidden_size)
        self.ln_kv = nn.LayerNorm(hidden_size)

        self.ln_final_decoder = nn.LayerNorm(hidden_size)

        self.cross_attn = TransformerDecoder(
            width=hidden_size,
            layers=layer_num,
            heads=head_num,
        )

        self.dev_convs = nn.ModuleList()
        cur_chn = hidden_size

        for _ in range(dev_convs_nums):
            self.dev_convs.append(nn.Sequential(
                nn.ConvTranspose2d(cur_chn, cur_chn // 2, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(cur_chn // 2),
                nn.GELU(),
            ))
            cur_chn = cur_chn // 2
            
        self.final_d = nn.Sequential(
            nn.Conv2d(cur_chn, cur_chn // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(cur_chn // 2),
            nn.GELU(),
            nn.Conv2d(cur_chn // 2, out_channels, kernel_size=1),
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # 使用 Kaiming 初始化卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                # 使用 Kaiming 初始化卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # 初始化批归一化层
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                # 初始化批归一化层
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        proj_std = (self.hidden_size ** -0.5) * ((2 * self.layer_num) ** -0.5)
        attn_std = self.hidden_size ** -0.5
        fc_std = (2 * self.hidden_size) ** -0.5

        
        nn.init.normal_(self.proj_q.weight, std=attn_std)
        nn.init.normal_(self.proj_kv.weight, std=attn_std)

        for block in self.cross_attn.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def att_image_to_text(self, encoded_image, llm_image):
        encoded_image = encoded_image.permute(1, 0, 2).contiguous()  # NLD -> LND
        llm_image = llm_image.permute(1, 0, 2).contiguous()

        tmp = self.cross_attn([llm_image, encoded_image, encoded_image, None])
        
        x = tmp[0]
        m = tmp[4]
        x = x.permute(1, 0, 2).contiguous() 
        x = self.ln_final_decoder(x)
        return x, m

    def forward(self, image_llm_hidden_features, image_encoder_attn_features, output_size, image_patch_size, image_token_num):

        image_llm_hidden_features = self.ln_q(self.proj_q(image_llm_hidden_features))

        image_encoder_attn_features = self.ln_kv(self.proj_kv(image_encoder_attn_features))

        image_embedding_features, attn_map = self.att_image_to_text(image_encoder_attn_features, image_llm_hidden_features)
                
        b, n, c = image_embedding_features.shape
        
        x = image_embedding_features.reshape(b * image_token_num, image_patch_size, image_patch_size, c).permute(0, 3, 1, 2)
        
        for i in range(self.dev_convs_nums):
            x = self.dev_convs[i](x)
        
        x = F.interpolate(x, size=(output_size, output_size), mode='nearest')

        x = self.final_d(x)
        
        return x, attn_map


