"""
input: [B, H, N, C]
output: [B, H, N, C]
"""
from time import time
from tqdm import tqdm
import argparse
import torch

parser = argparse.ArgumentParser(description='Test FlashAttention')
parser.add_argument('--attn_type', type=str, help='Attention type', default='dotscale', choices=['dotscale', 'flash_attn', 'imp_flash_attn'])
parser.add_argument('--epochs', type=int, help='Epoch num', default=1000)
parser.add_argument('--seq_len', type=int, help='Sequence length', default=-1)
args = parser.parse_args()

if args.attn_type == 'dotscale':
    from model import attention 
elif args.attn_type == 'flash_attn':
    from model_v1 import flash_attention_v1 
elif args.attn_type == 'imp_flash_attn':
    from model_v2 import flash_attention_v2 
    

if args.seq_len == -1:
    BHLC = [
        [32, 8, 200, 64],
        [32, 8, 400, 64],
        [32, 8, 800, 64],
        [32, 8, 1000, 64],
        [32, 8, 2000, 64],
    ]

else:
    BHLC = [
        [32, 8, args.seq_len, 64]
    ]

for shape in BHLC:
    B, H, L, C = shape
    if args.attn_type == 'dotscale':
        query = torch.randn(shape).cuda().to(torch.bfloat16)
        key = torch.randn(shape).cuda().to(torch.bfloat16)
        value = torch.randn(shape).cuda().to(torch.bfloat16)
    elif args.attn_type == 'flash_attn':
        from flash_attn.flash_attn_interface import flash_attn_unpadded_func
        
        query = torch.randn(B*L, H, C).cuda().to(torch.bfloat16)
        key = torch.randn(B*L, H, C).cuda().to(torch.bfloat16)
        value = torch.randn(B*L, H, C).cuda().to(torch.bfloat16)
        cu_seqlens_q = torch.arange(0, (B+1)*L, L, dtype=torch.int32).cuda()
        cu_seqlens_kv = torch.arange(0, (B+1)*L, L, dtype=torch.int32).cuda()
    elif args.attn_type == 'imp_flash_attn':
        from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
        
        query = torch.randn(B, L, H, C).cuda().to(torch.bfloat16)
        key = torch.randn(B, L, H, C).cuda().to(torch.bfloat16)
        value = torch.randn(B, L, H, C).cuda().to(torch.bfloat16)
    
    start_t = time()
    
    for e in tqdm(range(args.epochs)):
        if args.attn_type == 'dotscale':
            attn_out, _ = attention(query, key, value)    
        elif args.attn_type == 'flash_attn':
            flash_out = flash_attn_unpadded_func(query, key, value, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q=L, max_seqlen_k=L, dropout_p=0.0) 
        elif args.attn_type == 'imp_flash_attn':                                          
            imp_flash_out = flash_attn_func(query, key, value)
    
    end_t = time()
    
    print("序列长度为 {} 的序列经过 {} 个 epoch 的 {} 注意力运算的总时间消耗为 {}.".format(shape[2], args.epochs, args.attn_type, end_t - start_t))






    



