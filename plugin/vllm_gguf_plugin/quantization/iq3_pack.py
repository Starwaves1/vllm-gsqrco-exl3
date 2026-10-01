# SPDX-License-Identifier: Apache-2.0
"""Lossless repack of IQ3_S / IQ3_XXS weight bytes into the order the owned int8
tensor-core kernel (csrc/lcpp_owned_iq3_mma.cu, packed variant) reads them.

Same bytes per 16-row tile, same information, bijective (unpack(pack(w)) == w).
The rows are grouped in tiles of 16; tile t's bytes stay where rows 16t..16t+15
were, as one 16-row x 1-block record per weight block ("tile-block", 16 x 110 B
for IQ3_S, 16 x 98 B for IQ3_XXS, both multiples of 16 B). Inside a tile-block:

  lane (g, t) of the mma (g = lane / 4, t = lane % 4) holds rows g and g+8 and,
  of every 32-value slice s, the 4-value words w = 2t and 2t+1:
    [0, 512)     int4 per lane: row g's grid-index bytes, byte 2s+e = word 2t+e of slice s
    [512, 1024)  int4 per lane: the same for row g+8
    IQ3_S [1024, 1664): per word the 5 bits above the grid index (sign nibble, then qh),
      as 5 words H0..H4 per lane (int4 H0..H3, then u32 H4). Word i of the lane's 8
      "X" words (i = 4R + c: row g+8R, slices 2c and 2c+1) has, in byte j, the 5 bits
      of word 2t + (j & 1) of slice 2c + (j >> 1). X0..X4 are the low 5 bits of
      H0..H4's bytes; X5..X7 are spread over the bytes' top 3 bits (see _S_HI).
    IQ3_XXS [1024, 1472): per slice the 7 stored sign bits of the lane's word pair,
      re-coded (bit 3 = parity of the pair's first nibble, see _xxs_recode), one
      byte per (row, slice): B0..B3 (row g slices 0-3, 4-7, row g+8 slices 0-3, 4-7)
      as uint2 (B0, B1), u32 B2, u16 = B3's bytes 0, 1; B3's bytes 2, 3 (14 bits)
      sit in bit 7 of those 14 bytes.
    then per g: u32 sub-scales (nibble s = slice s) of rows g, g+8 (64 B),
    then per g: half d of rows g, g+8 (32 B).

The kernel then loads each lane's bytes with straight coalesced loads and turns a
word into its signed-grid table index with one byte permute (IQ3_S) or permute +
mask (IQ3_XXS): no qh / sign / parity unpacking per word.
"""

import torch
from gguf import GGML_QUANT_SIZES
from gguf import GGMLQuantizationType as WeightType

ROWS = 16  # rows per tile (the mma's M)
CHUNK_BYTES = 64 << 20  # pack_'s scratch cap

# IQ3_S: where X5..X7 (row g+8, slice pairs 1..3) live in H0..H4's top 3 bits, as
# (x, bit offset in x, h, bit offset in h's byte, width)
_S_HI = [(5, 0, 0, 5, 3), (5, 3, 1, 5, 2), (6, 0, 2, 5, 3), (6, 3, 3, 5, 2),
         (7, 0, 4, 5, 3), (7, 3, 1, 7, 1), (7, 4, 3, 7, 1)]


def _par(x: torch.Tensor, bits: int) -> torch.Tensor:
    p = torch.zeros_like(x)
    for i in range(bits):
        p ^= (x >> i) & 1
    return p


def _xxs_recode(raw: torch.Tensor) -> torch.Tensor:
    """7 sign bits (4 of the pair's first word, 3 of its second; the second word's
    4th sign is the parity of all 7) -> bits 0-2 the first word's signs 0-2, bit 3
    the parity of the first word's 4 signs, bits 4-6 the second word's signs 0-2.
    The kernel's table reads bits 0-3 for the first word and 3-6 for the second
    and rebuilds each word's 4th sign from its own nibble. Bijective on 7 bits."""
    return (raw & 7) | (_par(raw & 0xF, 4) << 3) | (raw & 0x70)


def _xxs_decode(b: torch.Tensor) -> torch.Tensor:
    e3 = ((b >> 3) & 1) ^ _par(b & 7, 3)
    return (b & 7) | (e3 << 3) | (b & 0x70)


def _lanes(v: torch.Tensor) -> torch.Tensor:
    """[N, R2, G8, S8, W8] per-word values -> [N, lane32, R2, 16] with byte 2s+e = word 2t+e."""
    n = v.shape[0]
    v = v.reshape(n, 2, 8, 8, 4, 2)                 # N R G S T E
    return v.permute(0, 2, 4, 1, 3, 5).reshape(n, 32, 2, 16)


def _pack_tiles(w: torch.Tensor, weight_type: WeightType) -> torch.Tensor:
    """pack() on whole tiles, in uint8. Reads strided views of w and writes each part of a
    tile-block straight into the output: scratch is the output plus up to ~2x w's bytes of
    per-word temporaries (IQ3_S's 5-bit words, IQ3_XXS's sign words)."""
    bsize = GGML_QUANT_SIZES[weight_type][1]
    rows, rb = w.shape
    nb = rb // bsize
    assert w.dtype == torch.uint8 and rows % ROWS == 0 and rb == nb * bsize, (w.shape, w.dtype)
    lead = (rows // ROWS, nb)                        # tile, block
    b = w.view(rows // ROWS, 2, 8, nb, bsize).permute(0, 3, 1, 2, 4)  # view: tile, block, R, G, bytes
    n = lead[0] * nb
    out = torch.empty_like(w)
    o = out.view(*lead, ROWS * bsize)                # tile t's bytes, block j at j * 16 * bsize
    fb = 1664 if weight_type == WeightType.IQ3_S else 1472
    # grid-index bytes: [R, lane (G, T), byte (S, E)] from word 2T+E of slice S
    q = b[..., 2:66].reshape(*lead, 2, 8, 8, 4, 2)   # R G S T E
    o[..., :1024].view(*lead, 2, 8, 4, 8, 2).copy_(q.permute(0, 1, 2, 3, 5, 4, 6))
    o[..., fb + 64:].view(*lead, 8, 2, 2).copy_(b[..., 0:2].transpose(2, 3))  # d per g: rows g, g+8
    u8 = dict(dtype=torch.uint8, device=w.device)
    if weight_type == WeightType.IQ3_S:
        o[..., fb:fb + 64].view(*lead, 8, 2, 4).copy_(b[..., 106:110].transpose(2, 3))
        # elementwise results keep b's permuted strides: reshape (not view) to N
        hn = (b[..., 74:106].reshape(*lead, 2, 8, 8, 4, 1) >> torch.tensor([0, 4], **u8)).reshape(n, 2, 8, 8, 8)
        hn &= 0xF                                    # N R G S W: sign nibble
        hn |= ((b[..., 66:74].reshape(*lead, 2, 8, 8, 1) >> torch.arange(8, **u8)) & 1).reshape(n, 2, 8, 8, 8) << 4
        x = _lanes(hn).reshape(n, 32, 8, 4)          # X_i byte j, i = 4R + c
        del hn
        h = x[:, :, :5].clone()
        for xi, xo, hi, ho, width in _S_HI:
            h[:, :, hi] |= ((x[:, :, xi] >> xo) & ((1 << width) - 1)) << ho
        o[..., 1024:1536].view(n, 32, 4, 4).copy_(h[:, :, :4])
        o[..., 1536:1664].view(n, 32, 4).copy_(h[:, :, 4])
    else:
        aux = b[..., 66:98].reshape(*lead, 2, 8, 8, 4).contiguous().view(torch.int32)[..., 0]  # R G S, LE
        sc4 = ((aux >> 28) & 0xF).to(torch.uint8)
        o[..., fb:fb + 64].view(*lead, 8, 2, 4).copy_((sc4[..., 0::2] | sc4[..., 1::2] << 4).transpose(2, 3))
        raw = torch.stack([((aux >> 7 * i) & 0x7F).to(torch.uint8) for i in range(4)], -1)  # R G S T
        del aux, sc4
        bb = _xxs_recode(raw).reshape(n, 2, 8, 8, 4).permute(0, 2, 4, 1, 3).reshape(n, 32, 16)  # lane, (R, S)
        del raw
        sp = bb[:, :, 14].short() | bb[:, :, 15].short() << 7  # B3 bytes 2, 3 -> bit 7 of bytes 0..13
        lo = bb[:, :, :14] | ((sp.unsqueeze(-1) >> torch.arange(14, dtype=torch.int16, device=w.device)) & 1).to(torch.uint8) << 7
        o[..., 1024:1280].view(n, 32, 8).copy_(lo[:, :, 0:8])
        o[..., 1280:1408].view(n, 32, 4).copy_(lo[:, :, 8:12])
        o[..., 1408:1472].view(n, 32, 2).copy_(lo[:, :, 12:14])
    return out


def pack_(w: torch.Tensor, weight_type: int) -> None:
    """pack() in place, a group of whole tiles at a time. _pack_tiles needs ~3x its chunk in
    scratch (2.96x measured on IQ3_XXS, less on IQ3_S), so chunks of min(w's bytes, CHUNK_BYTES) / 4
    keep the scratch at ~0.75x of that. Below 64 rows the chunk is one 16-row tile (~3x w)."""
    weight_type = WeightType(weight_type)
    assert weight_type in (WeightType.IQ3_S, WeightType.IQ3_XXS), weight_type
    rows, rb = w.shape
    assert rows % ROWS == 0, w.shape
    step = ROWS * max(1, min(CHUNK_BYTES, w.numel()) // (4 * ROWS * rb))
    for r0 in range(0, rows, step):
        w[r0:r0 + step].copy_(_pack_tiles(w[r0:r0 + step], weight_type))


def pack(w: torch.Tensor, weight_type: int) -> torch.Tensor:
    """[rows, row_bytes] uint8 GGUF blocks (rows % 16 == 0) -> the packed bytes, same shape."""
    p = w.clone()
    pack_(p, weight_type)
    return p


def unpack(p: torch.Tensor, weight_type: int) -> torch.Tensor:
    """Inverse of pack: the GGUF block bytes."""
    weight_type = WeightType(weight_type)
    assert weight_type in (WeightType.IQ3_S, WeightType.IQ3_XXS), weight_type
    bsize = GGML_QUANT_SIZES[weight_type][1]
    rows, rb = p.shape
    nb = rb // bsize
    t = p.reshape(-1, ROWS * bsize).int()
    n = t.shape[0]
    fb = 1664 if weight_type == WeightType.IQ3_S else 1472

    def from_lanes(v):  # [N, lane32, R2, 16] -> [N, R, G, S, W]
        v = v.reshape(n, 8, 4, 2, 8, 2)             # N G T R S E
        return v.permute(0, 3, 1, 4, 2, 5).reshape(n, 2, 8, 8, 8)

    q = from_lanes(torch.stack([t[:, 0:512].reshape(n, 32, 16), t[:, 512:1024].reshape(n, 32, 16)], 2))
    sc = t[:, fb:fb + 64].reshape(n, 8, 2, 4).permute(0, 2, 1, 3)       # N R G 4
    d = t[:, fb + 64:fb + 96].reshape(n, 8, 2, 2).permute(0, 2, 1, 3)
    if weight_type == WeightType.IQ3_S:
        h = torch.cat([t[:, 1024:1536].reshape(n, 32, 4, 4), t[:, 1536:1664].reshape(n, 32, 1, 4)], 2)
        x = torch.cat([h & 0x1F, torch.zeros_like(h[:, :, :3])], 2)
        for xi, xo, hi, ho, width in _S_HI:
            x[:, :, xi] |= ((h[:, :, hi] >> ho) & ((1 << width) - 1)) << xo
        hn = from_lanes(x.reshape(n, 32, 2, 16))
        qh = ((hn >> 4) << torch.arange(8, device=p.device)).sum(-1)       # N R G S
        nib = hn & 0xF
        signs = (nib[..., 0::2] | nib[..., 1::2] << 4).reshape(n, 2, 8, 32)
        body = [q.reshape(n, 2, 8, 64), qh, signs, sc]
    else:
        lo = torch.cat([t[:, 1024:1280].reshape(n, 32, 8), t[:, 1280:1408].reshape(n, 32, 4),
                        t[:, 1408:1472].reshape(n, 32, 2)], 2)
        sp = ((lo >> 7) << torch.arange(14, device=p.device)).sum(-1)
        bb = torch.cat([lo & 0x7F, (sp & 0x7F).unsqueeze(-1), (sp >> 7).unsqueeze(-1)], 2)
        raw = _xxs_decode(bb.reshape(n, 8, 4, 2, 8).permute(0, 3, 1, 4, 2))  # N R G S T
        sc4 = torch.stack([sc[..., i // 2] >> (4 * (i % 2)) & 0xF for i in range(8)], -1)  # N R G S
        aux = (raw << (7 * torch.arange(4, device=p.device))).sum(-1) | sc4 << 28
        aux = torch.stack([(aux >> (8 * i)) & 0xFF for i in range(4)], -1).reshape(n, 2, 8, 32)
        body = [q.reshape(n, 2, 8, 64), aux]
    blocks = torch.cat([d] + body, -1)               # N R G bsize
    return (blocks.reshape(rows // ROWS, nb, ROWS, bsize).permute(0, 2, 1, 3)
            .reshape(rows, rb).to(torch.uint8))
