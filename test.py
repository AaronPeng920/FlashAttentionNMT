"""
input: [B, H, N, C]
output: [B, H, N, C]
"""
from time import time
from tqdm import tqdm
import torch

from model import attention

BHLC = [
    [32, 8, 200, 64],
    [32, 8, 400, 64],
    [32, 8, 800, 64],
    [32, 8, 1000, 64],
    [32, 8, 2000, 64],
]


type = 'flash_v2'
epoch = 10000


for shape in BHLC:
    B, H, L, C = shape
    if type == 'attn':
        query = torch.randn(shape).cuda().to(torch.bfloat16)
        key = torch.randn(shape).cuda().to(torch.bfloat16)
        value = torch.randn(shape).cuda().to(torch.bfloat16)
    elif type == 'flash_v1':
        from flash_attn.flash_attn_interface import flash_attn_unpadded_func
        
        query = torch.randn(B*L, H, C).cuda().to(torch.bfloat16)
        key = torch.randn(B*L, H, C).cuda().to(torch.bfloat16)
        value = torch.randn(B*L, H, C).cuda().to(torch.bfloat16)
        cu_seqlens_q = torch.arange(0, (B+1)*L, L, dtype=torch.int32).cuda()
        cu_seqlens_kv = torch.arange(0, (B+1)*L, L, dtype=torch.int32).cuda()
    elif type == 'flash_v2':
        from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
        
        query = torch.randn(B, L, H, C).cuda().to(torch.bfloat16)
        key = torch.randn(B, L, H, C).cuda().to(torch.bfloat16)
        value = torch.randn(B, L, H, C).cuda().to(torch.bfloat16)
    
    start_t = time()
    
    for e in tqdm(range(epoch)):
        if type == 'attn':
            attn_out, _ = attention(query, key, value)    
        elif type == 'flash_v1':
            flash_v1_out = flash_attn_unpadded_func(query, key, value, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q=L, max_seqlen_k=L, dropout_p=0.0) 
        elif type == 'flash_v2':                                          
            flash_v2_out = flash_attn_func(query, key, value)
    
    end_t = time()
    
    print(end_t - start_t)






    



