"""Transformer encoder classifier for the majority-token task
(code/transformers/majority_token_dataset.py).

Architecture, token embedding -> CLS + positional embedding -> stack of
TransformerEncoderBlock (code/transformers/transformer_block.py, already
containing multi-head self-attention, residual connections, LayerNorm,
and a position-wise MLP) -> classification head on the CLS token. This
is the standard "BERT-style" encoder-only classification recipe (a
learned [CLS] token whose final-layer representation is used for
classification), applied here to a non-image, non-text sequence so the
Transformer itself is exercised before specializing it to image patches
(ViT, a later exercise).

Symbol -> code variable -> shape -> meaning:
    L (seq_len)              seq_len            scalar       input sequence length (excludes CLS)
    V (vocab_size)            vocab_size         scalar       number of distinct token values = number of classes
    tokens                    tokens             [B,L]        int64 input token ids, values in [0,V)
    token embedding table     token_embedding    [V,D]        nn.Embedding weight
    E (token embeds)          token_embeds       [B,L,D]      token_embedding(tokens)
    CLS embedding             cls_token           [1,1,D]      learned, broadcast to every batch element
    X (tokens with CLS)       x                   [B,L+1,D]    cls_token concatenated in front of token_embeds
    positional embedding      position_embedding [1,L+1,D]    learned, one vector per position (0 = CLS)
    encoder stack output      x (after blocks)    [B,L+1,D]    shape-preserving stack of TransformerEncoderBlock
    CLS output                cls_output          [B,D]        x[:,0,:], the position-0 (CLS) token after all blocks
    classifier weights        classifier_head     [D,V]        nn.Linear(D -> num_classes)
    logits                     logits              [B,V]        unnormalized class scores

Indexing convention: position 0 along the token axis is always the CLS
token (prepended, never one of the L data tokens), so position_embedding
index 0 belongs to CLS and indices 1..L belong to data tokens 0..L-1 in
order; this fixed convention is what lets position_embedding be a plain
learned [1,L+1,D] tensor added by broadcasting rather than looked up.

Local usage (lightweight, CPU, tiny tensors, no dataset, no GPU):
    python code/transformers/majority_token_classifier.py --smoke-test
"""

import argparse

import torch
import torch.nn as nn

from transformer_block import TransformerEncoderBlock


class MajorityTokenTransformerClassifier(nn.Module):
    """Encoder-only Transformer classifier: Embedding -> [CLS; tokens] +
    positional embedding -> N x TransformerEncoderBlock -> Linear(CLS)."""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ffn_hidden_dim: int,
        num_classes: int | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_classes = num_classes if num_classes is not None else vocab_size

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)  # [V,D]
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)  # [1,1,D]
        self.position_embedding = nn.Parameter(torch.randn(1, seq_len + 1, embed_dim) * 0.02)  # [1,L+1,D]

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, num_heads, ffn_hidden_dim)
                for _ in range(num_layers)
            ]
        )
        self.classifier_head = nn.Linear(embed_dim, self.num_classes)  # [D,V]

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            tokens: [B,L] int64, L == self.seq_len, values in [0, vocab_size)

        Returns:
            logits:           [B,num_classes]
            cls_output:        [B,D], final-layer CLS representation
            x:                 [B,L+1,D], final-layer all-token representation
            attn_weights:      list of length num_layers, each [B,heads,L+1,L+1],
                               one entry per encoder block (last layer's entry
                               is what majority_token_evaluate.py visualizes)
        """
        batch_size, seq_len = tokens.shape
        assert seq_len == self.seq_len

        token_embeds = self.token_embedding(tokens)  # [B,L] -> [B,L,D]

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [1,1,D] -> [B,1,D], broadcast per batch element
        x = torch.cat([cls_tokens, token_embeds], dim=1)  # [B,1,D]+[B,L,D] -> [B,L+1,D], CLS prepended at position 0

        x = x + self.position_embedding  # [B,L+1,D] + [1,L+1,D] -> [B,L+1,D], broadcast over batch

        attn_weights = []
        for block in self.blocks:
            intermediates = block(x)  # dict, see TransformerEncoderBlock.forward
            x = intermediates["z"]  # [B,L+1,D], shape-preserving
            attn_weights.append(intermediates["attn_attention_weights"])  # [B,heads,L+1,L+1]

        cls_output = x[:, 0, :]  # [B,L+1,D] -> [B,D], CLS = position 0 after all blocks
        logits = self.classifier_head(cls_output)  # [B,D] -> [B,num_classes]

        return {
            "logits": logits,
            "cls_output": cls_output,
            "x": x,
            "attn_weights": attn_weights,
        }


def print_classifier_shapes(tokens: torch.Tensor, outputs: dict) -> None:
    """Prints the shape at every major stage, from raw token ids to class logits."""
    print(f"tokens (input)                      : {tuple(tokens.shape)}")
    print(f"x (after CLS + positional embedding) : {tuple(outputs['x'].shape)}")
    print(f"attn_weights[-1] (last-layer attn)   : {tuple(outputs['attn_weights'][-1].shape)}")
    print(f"cls_output (= x[:,0,:])              : {tuple(outputs['cls_output'].shape)}")
    print(f"logits                               : {tuple(outputs['logits'].shape)}")


def _smoke_test():
    """Lightweight local checks: shapes end-to-end, finite logits, cross-
    entropy loss + backward reaches every parameter (including the CLS
    token and positional embedding), and attention rows still sum to 1
    inside the full classifier (not just the standalone block)."""
    torch.manual_seed(0)
    vocab_size, seq_len = 7, 6
    embed_dim, num_heads, num_layers, ffn_hidden_dim = 8, 2, 2, 16
    batch_size = 4

    model = MajorityTokenTransformerClassifier(
        vocab_size, seq_len, embed_dim, num_heads, num_layers, ffn_hidden_dim
    )
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size,))

    outputs = model(tokens)

    assert outputs["x"].shape == (batch_size, seq_len + 1, embed_dim)
    assert len(outputs["attn_weights"]) == num_layers
    assert outputs["attn_weights"][-1].shape == (batch_size, num_heads, seq_len + 1, seq_len + 1)
    assert outputs["cls_output"].shape == (batch_size, embed_dim)
    assert outputs["logits"].shape == (batch_size, vocab_size)
    assert torch.isfinite(outputs["logits"]).all()

    row_sums = outputs["attn_weights"][-1].sum(dim=-1)  # [B,heads,L+1]
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    loss = nn.functional.cross_entropy(outputs["logits"], labels)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} gradient is not finite"

    print("majority_token_classifier smoke test passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="run lightweight local checks and exit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, default=12)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ffn-hidden-dim", type=int, default=128)
    args = parser.parse_args()

    if args.smoke_test:
        _smoke_test()
    else:
        torch.manual_seed(args.seed)
        model = MajorityTokenTransformerClassifier(
            args.vocab_size, args.seq_len, args.embed_dim, args.num_heads, args.num_layers, args.ffn_hidden_dim
        )
        tokens = torch.randint(0, args.vocab_size, (2, args.seq_len))
        outputs = model(tokens)
        print_classifier_shapes(tokens, outputs)
