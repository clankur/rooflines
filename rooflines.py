# %%
from collections import namedtuple

AcceleratorSpec = namedtuple('AcceleratorSpec', ['bf16_flops', 'hbm_bw', 'ici_bw'])
# bf16_flops: peak bf16 FLOPS (bytes/sec), 
# hbm_bw: HBM memory bandwidth (bytes/sec), 
# ici_bw: inter-chip interconnect bandwidth (bytes/sec)

ACCELERATORS = {
    'h100': AcceleratorSpec(bf16_flops=9.89e14, hbm_bw=3.35e12, ici_bw=0.9e12),
    'v6e' : AcceleratorSpec(bf16_flops=9.1e14,  hbm_bw=1.76e12, ici_bw=0.8e12),
}

# %%
BATCH_SIZE = 256
N_ACTIVE = 1e12
L = 4e6
TOKEN_BYTES_BF16 = 2

# %%
chip = ACCELERATORS['h100']
flops =  chip.bf16_flops
hbm_bandwidth = chip.hbm_bw

# %% 
# number of floating point operations 
# effectively mat muls = floating point mul + add operations

# for simplicity this is 
computation_flops = BATCH_SIZE * N_ACTIVE * 2
# notes:
# - this drops attention for simplicity (the ffn is dominating factor) *at short context lengths*
# - in the lecture reiner does not including the number of floating point ops for 
#   mul + add ie). the 2 at the end we have
# - if we  want to be pedantic:
# ? what is n and k going to refer to in attention
#   k = K_len maybe?
#   n = hidden dim maybe
# given an pair of vectors of size k, we do
#   einsum(k, k -> 1)  
# does k muls, k - 1 adds
# meaning we have been undercounting adds

# so for a dot product  
# => given an vector of size k and a tensor of shape (n, k), we do 
#    einsum(k, k n -> n) 
# does k * n muls, (k - 1) * n adds
# now lets actually make some estimates
# N_W_PARAMS: (k n) = k * n 
# flops per vector / token (k): 
#   mul_flops: k * n
#   add_flops: (k-1) * n
#   approximately: 2 * k * n = 2 * N_W_PARAMS
#
# projection flops ~= 2 * N_W_PARAMS
# if projections are sparse
# activated_ratio = NUM_SELECTED / TOTAL_EXPERTS
# projection flops ~= 2 * N_W_PARAMS * activated_ratio
# projection flops from ffn ~= 2 * (M * F) * activated_ratio
#   up: K is M, N is F
#   mul_flops: M * F
#   add_flops: (M-1) * F
#   down: K is M, N is F
#   mul_flops: (F-1) * M
#   down_flops: F * M

# TLDR: just understand we are simpllifying since its easier to keep track of
# the approximation makes all computations easier to track

# %%
t_math = computation_flops / flops

# %%
communication_bytes = N_ACTIVE + BATCH_SIZE * L * TOKEN_BYTES_BF16

# %%
t_memory = communication_bytes / hbm_bandwidth 
t_memory
# %%
t_lower_chip = max(t_math, t_memory)
# whoever is the domination component will be what we are bound by
t_upper_chip = t_math + t_memory
t_lower_chip, t_upper_chip
# t_upper_chip <= 2 * t_lower_chip
# %%
# determining what we are bound by
op_intensity = lambda computation_flops, communication_bytes: computation_flops / communication_bytes
# unit = FLOPs / byte
# t_math > t_memory 
# <=> computation_flops / flops > communication_bytes / hbm_bandwidth
# <=> computation_flops / communication_bytes > flops / hbm_bandwidth
# <=> op_intensity(computation) > op_intensity(chip)
# the quantity op_intensity(chip) = the arithmetic intensity where the chip achieves peak FLOPs/sec
op_intensity_chip = flops / hbm_bandwidth 
op_intensity_chip
# %%
# dot product roofline
# x: [N], y: [N]
# x • y: bf16[N], bf16[N] → bf16[1]
# each x, y load from memory = 2 * N 
# *also write back* 2 bytes into memory for the output
# communication_bytes = 2 * 2 * N + 2 = 4 * N + 2
# as mentioned earlier = dot product does N muls, N - 1 adds 
# computations_flops = 2 * N - 1
# computation = (computations_flops, communication_bytes)
# op_intensity(computation)
# lim N -> inf = 1/2
# the dot product 0.5 FLOP per byte loaded

# %%
# X: [B D], Y: [D F]
# X • Y: bf16[B D], bf16[D F] → bf16[B F]
# each X, Y load from memory = 2 * (B * D + D * F)
# also write back  = B * F * 2
# communication_bytes = 2 * (B * D + D * F + B * F)
# mat mul does B * F dot products
    # each is D muls, ~D adds
# computations_flops = 2 * B * D * F
# computation = (computations_flops, communication_bytes)
# op_intensity(computation) 
# (B * D * F) / (B * D + D * F + B * F)
# assume B is small relative to D and F
# <=> (B * D * F) / (D * F) = B
# <=> op_intensity(computation) > op_intensity(chip)
# <=> B > op_intensity(chip)
# general rule of thumb is to have batch size > operational intensity of your chip

# %%
# Network rooflines
# X: [B D], Y: [D F]
# evenly split across 2 chips
#   sharded along D
# A = X[:, :D // 2] @ Y[:D // 2, :]
# B = X[:, D // 2:] @ Y[D // 2:, :]
# computation_flops = 2 * B * D * F
# t_math = computation_flops / flops
# t_comms now refers to the communciation between chips
# ie). total bytes sent over ici_bandwitch
# communication_bytes = 2 * B * F
# t_comms = communication_bytes  / ici_bandwitch

# we are compute bound when
# op_intensity(computation_flops, communication_bytes) > op_intesity(flops, ici_bandwidth)
# <=> BDF / 2BF > flops / ici_bandwidth
# <=> D / 2 > flops / ici_bandwidth
# D > flops / ici_bandwidth * 2

# %%
# 