"""Educational Transformer encoder block: multi-head self-attention,
residual connections, LayerNorm, and a position-wise MLP.

Paper: Vaswani et al., 2017, "Attention Is All You Need",
Sec. 3.1 (encoder sub-layer wrapping), Sec. 3.2.2 (multi-head attention),
Sec. 3.3 (position-wise feed-forward network).

One encoder layer has two sub-layers, each wrapped the same way (the
paper's literal post-LN wrapping, Sec. 3.1):

    sub-layer output = LayerNorm(x + Sublayer(x))

Sub-layer 1, multi-head self-attention (Q=K=V=X):

    MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O
    head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
           = softmax(Q_i K_i^T / sqrt(d_k)) V_i
    d_k = d_v = D / h

Sub-layer 2, position-wise feed-forward network:

    FFN(y) = max(0, y W_1 + b_1) W_2 + b_2

Full block (X: input, Z: output, both [B,N,D] -- the block is shape-
preserving):

    attn_out = MultiHead(X, X, X)          # [B,N,D]
    Y        = LayerNorm(X + attn_out)     # [B,N,D], sub-layer 1 residual + norm
    ffn_out  = FFN(Y)                      # [B,N,D]
    Z        = LayerNorm(Y + ffn_out)      # [B,N,D], sub-layer 2 residual + norm = block output

Symbol -> code variable -> shape -> meaning:
    X                      x                       [B,N,D]     block input tokens (B=batch, N=#tokens, D=embed/model dim)
    h                      num_heads               scalar      number of attention heads
    d_k, d_v               head_dim                scalar      D // h (per-head query/key/value dim)
    W_i^Q, W_i^K, W_i^V    (combined) W_q,W_k,W_v   [D,D]       all heads' projections stacked, no bias (matches Eq. 1/multi-head, same convention as self_attention_qkv.py)
    Q, K, V (split)        q, k, v                 [B,h,N,d_k] per-head projected queries/keys/values
    Q K^T (per head)       raw_scores              [B,h,N,N]   raw_scores[b,head,i,j] = q[b,head,i] . k[b,head,j]
    softmax(.)             attention_weights       [B,h,N,N]   row i sums to 1 per head: how much query i attends to key j
    softmax(.) V           head_outputs            [B,h,N,d_k] per-head weighted sums of V
    Concat(heads)          context                 [B,N,D]     heads merged back along the feature axis
    W^O                    W_o                     [D,D]       output mixing projection, no bias
    MultiHead(X,X,X)       attn_out                [B,N,D]     multi-head self-attention output
    LayerNorm(X+attn_out)  y                       [B,N,D]     sub-layer 1 residual + LayerNorm
    W_1, b_1               ffn_linear1             [D,d_ff]    FFN expansion
    W_2, b_2               ffn_linear2             [d_ff,D]    FFN projection back to D
    FFN(y)                 ffn_out                 [B,N,D]     position-wise feed-forward output
    LayerNorm(y+ffn_out)   z                       [B,N,D]     sub-layer 2 residual + LayerNorm = block output

Indexing convention: heads are a batched leading axis, exactly like the
batch axis B -- torch.matmul on [B,h,N,d_k] @ [B,h,d_k,N] batches over
both B and h independently, contracting only the trailing d_k axis.
Splitting [B,N,D] into [B,h,N,d_k] and merging back is a pure reshape
of the same D = h * d_k values; no information is created or discarded.

This file keeps the same "manual" style as self_attention_qkv.py: Q/K/V
projections and the two Linear layers of the FFN are nn.Linear (standard
and not worth reinventing), but the per-head scaled dot product, the
softmax (via manual_softmax, imported from self_attention_qkv.py), and
the head merge are written out explicitly -- no nn.MultiheadAttention,
no nn.TransformerEncoderLayer.

Local usage (lightweight, CPU, no dataset, no GPU):
    python code/transformers/transformer_block.py --smoke-test
"""

import argparse

import torch
import torch.nn as nn

from self_attention_qkv import manual_softmax, make_toy_token_sequence


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention, Sec. 3.2.2 of Vaswani et al., 2017,
    applied as self-attention (Q, K, V all derived from the same input X)."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads (d_k = embed_dim / num_heads)"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Combined per-head projections for all heads at once: [D,D] each.
        # No bias, matching Eq. 1 / the paper's multi-head formula exactly.
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)  # output mixing projection W^O

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        """[B,N,D] -> [B,h,N,d_k]: split the feature axis D into (h,d_k),
        then move h in front of N so heads behave as an extra batched
        axis for torch.matmul, exactly like the existing B axis."""
        batch_size, num_tokens, embed_dim = t.shape
        t = t.view(batch_size, num_tokens, self.num_heads, self.head_dim)  # [B,N,D] -> [B,N,h,d_k], pure reshape of D=h*d_k
        t = t.transpose(1, 2)  # [B,N,h,d_k] -> [B,h,N,d_k], move heads next to batch
        return t

    def _merge_heads(self, t: torch.Tensor) -> torch.Tensor:
        """[B,h,N,d_k] -> [B,N,D]: inverse of _split_heads. transpose(1,2)
        returns a non-contiguous view, so .contiguous() is required before
        .view() can reinterpret the memory as [B,N,D] (view refuses to
        silently copy; reshape would, but .contiguous()+.view() keeps the
        copy explicit)."""
        batch_size, num_heads, num_tokens, head_dim = t.shape
        t = t.transpose(1, 2)  # [B,h,N,d_k] -> [B,N,h,d_k]
        t = t.contiguous().view(batch_size, num_tokens, num_heads * head_dim)  # [B,N,h,d_k] -> [B,N,D]
        return t

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: [B,N,D] input token embeddings, D == self.embed_dim

        Returns a dict of every named intermediate tensor:
            q, k, v:            [B,h,N,d_k]
            raw_scores:         [B,h,N,N], Q K^T before scaling
            scaled_scores:      [B,h,N,N], raw_scores / sqrt(d_k)
            attention_weights:  [B,h,N,N], softmax(scaled_scores), rows sum to 1
            head_outputs:       [B,h,N,d_k], per-head softmax(.) V
            context:            [B,N,D], heads merged back together
            output:             [B,N,D], context @ W^O
        """
        batch_size, num_tokens, embed_dim = x.shape
        assert embed_dim == self.embed_dim

        q = self._split_heads(self.W_q(x))  # [B,N,D] -> [B,N,D] -> [B,h,N,d_k]
        k = self._split_heads(self.W_k(x))  # [B,h,N,d_k]
        v = self._split_heads(self.W_v(x))  # [B,h,N,d_k]

        # Per head (and per batch element), Q K^T: [B,h,N,d_k] @ [B,h,d_k,N] -> [B,h,N,N].
        # transpose(-2,-1) swaps the last two axes so matmul contracts over
        # d_k (shared feature axis), batching independently over both B and h.
        raw_scores = torch.matmul(q, k.transpose(-2, -1))  # [B,h,N,d_k] @ [B,h,d_k,N] -> [B,h,N,N]

        scale = self.head_dim ** 0.5  # sqrt(d_k), scalar
        scaled_scores = raw_scores / scale  # [B,h,N,N], unchanged shape

        attention_weights = manual_softmax(scaled_scores, dim=-1)  # [B,h,N,N], softmax over keys (last axis)

        # attention_weights @ v: [B,h,N,N] @ [B,h,N,d_k] -> [B,h,N,d_k];
        # contracts over the key axis N (3rd axis), batched over B and h.
        head_outputs = torch.matmul(attention_weights, v)  # [B,h,N,N] @ [B,h,N,d_k] -> [B,h,N,d_k]

        context = self._merge_heads(head_outputs)  # [B,h,N,d_k] -> [B,N,D], Concat(head_1,...,head_h)
        output = self.W_o(context)  # [B,N,D] -> [B,N,D], final W^O mixing

        return {
            "q": q,
            "k": k,
            "v": v,
            "raw_scores": raw_scores,
            "scaled_scores": scaled_scores,
            "attention_weights": attention_weights,
            "head_outputs": head_outputs,
            "context": context,
            "output": output,
        }


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network, Sec. 3.3: FFN(y) = max(0, y W_1 + b_1) W_2 + b_2.
    Applied identically (same weights) to every token position -- the
    "position-wise" in the name means no mixing across the N axis here,
    unlike attention."""

    def __init__(self, embed_dim: int, ffn_hidden_dim: int):
        super().__init__()
        self.ffn_linear1 = nn.Linear(embed_dim, ffn_hidden_dim)  # W_1,b_1: [D,d_ff]
        self.relu = nn.ReLU()
        self.ffn_linear2 = nn.Linear(ffn_hidden_dim, embed_dim)  # W_2,b_2: [d_ff,D]

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: [B,N,D]
        Returns:
            ffn_out: [B,N,D]
        """
        hidden = self.relu(self.ffn_linear1(y))  # [B,N,D] -> [B,N,d_ff]
        ffn_out = self.ffn_linear2(hidden)  # [B,N,d_ff] -> [B,N,D]
        return ffn_out


class TransformerEncoderBlock(nn.Module):
    """One Transformer encoder block (Sec. 3.1): multi-head self-attention
    and a position-wise FFN, each wrapped in a residual connection followed
    by LayerNorm (post-LN, as literally specified by the paper). Shape-
    preserving: output shape == input shape == [B,N,D]."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_hidden_dim: int):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)  # normalizes over the last axis D, independently per (b,n)
        self.feed_forward = PositionwiseFeedForward(embed_dim, ffn_hidden_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: [B,N,D]
        Returns a dict of every named intermediate tensor, in the order
        they are produced:
            attn_out:  [B,N,D], MultiHead(X,X,X)
            y:         [B,N,D], LayerNorm(X + attn_out), sub-layer 1 output
            ffn_out:   [B,N,D], FFN(y)
            z:         [B,N,D], LayerNorm(y + ffn_out), block output
        plus every key from MultiHeadSelfAttention.forward under "attn_".
        """
        attn_intermediates = self.self_attention(x)
        attn_out = attn_intermediates["output"]  # [B,N,D]

        y = self.norm1(x + attn_out)  # [B,N,D] + [B,N,D] -> [B,N,D], then LayerNorm over D: sub-layer 1 residual + norm

        ffn_out = self.feed_forward(y)  # [B,N,D]

        z = self.norm2(y + ffn_out)  # [B,N,D] + [B,N,D] -> [B,N,D], then LayerNorm over D: sub-layer 2 residual + norm

        result = {f"attn_{name}": tensor for name, tensor in attn_intermediates.items()}
        result.update({
            "attn_out": attn_out,
            "y": y,
            "ffn_out": ffn_out,
            "z": z,
        })
        return result


def print_block_shapes(x: torch.Tensor, intermediates: dict[str, torch.Tensor]) -> None:
    """Prints the [B,N,D] shape at each major step of the block, to make
    the shape-preserving property (input shape == output shape) explicit."""
    print(f"x (block input)                : {tuple(x.shape)}")
    print(f"attn_out (= MultiHead(X,X,X))   : {tuple(intermediates['attn_out'].shape)}")
    print(f"y (= LayerNorm(X + attn_out))   : {tuple(intermediates['y'].shape)}")
    print(f"ffn_out (= FFN(y))              : {tuple(intermediates['ffn_out'].shape)}")
    print(f"z (= LayerNorm(y + ffn_out))    : {tuple(intermediates['z'].shape)}")
    print(f"shape preserved (x.shape == z.shape): {tuple(x.shape) == tuple(intermediates['z'].shape)}")


def _smoke_test():
    """Lightweight local checks: shapes at every step (including the
    split/merge head reshape round trip), per-head attention rows sum to
    1, finite values, gradients flow to every parameter, fixed-seed
    reproducibility, and input/output shape equality. No dataset, no GPU."""
    torch.manual_seed(0)
    batch_size, num_tokens, embed_dim, num_heads, ffn_hidden_dim = 2, 5, 8, 2, 16
    head_dim = embed_dim // num_heads

    x = torch.randn(batch_size, num_tokens, embed_dim, requires_grad=True)
    block = TransformerEncoderBlock(embed_dim, num_heads, ffn_hidden_dim)

    intermediates = block(x)

    assert intermediates["attn_q"].shape == (batch_size, num_heads, num_tokens, head_dim)
    assert intermediates["attn_attention_weights"].shape == (batch_size, num_heads, num_tokens, num_tokens)
    assert intermediates["attn_context"].shape == (batch_size, num_tokens, embed_dim)
    assert intermediates["attn_out"].shape == (batch_size, num_tokens, embed_dim)
    assert intermediates["y"].shape == (batch_size, num_tokens, embed_dim)
    assert intermediates["ffn_out"].shape == (batch_size, num_tokens, embed_dim)
    assert intermediates["z"].shape == (batch_size, num_tokens, embed_dim)

    # Block is shape-preserving.
    assert intermediates["z"].shape == x.shape

    # Per-head attention rows are probability distributions over keys.
    row_sums = intermediates["attn_attention_weights"].sum(dim=-1)  # [B,h,N]
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    assert torch.isfinite(intermediates["z"]).all()

    # Gradients must flow back to every learnable parameter.
    loss = intermediates["z"].sum()
    loss.backward()
    for name, param in block.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} gradient is not finite"

    # Fixed-seed reproducibility: same module (no randomness in forward), same input -> identical output.
    with torch.no_grad():
        z_repeat = block(x)["z"]
    assert torch.equal(intermediates["z"], z_repeat)

    print("\ntransformer_block smoke test passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="run lightweight local checks and exit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--ffn-hidden-dim", type=int, default=16)
    args = parser.parse_args()

    if args.smoke_test:
        _smoke_test()
    else:
        torch.manual_seed(args.seed)
        x, tokens = make_toy_token_sequence(seed=args.seed)  # x: [1,N,8]
        embed_dim = x.shape[-1]

        block = TransformerEncoderBlock(embed_dim, args.num_heads, args.ffn_hidden_dim)
        intermediates = block(x)

        print(f"tokens = {tokens}")
        print_block_shapes(x, intermediates)
