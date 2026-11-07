"""Educational minimal Vision Transformer (ViT) patch embedding + encoder.

Paper: Dosovitskiy et al., 2021, "An Image Is Worth 16x16 Words:
Transformers for Image Recognition at Scale", Eq. 1-4.

An image is split into fixed-size patches, each patch is flattened and
linearly projected to an embedding dimension D (exactly like a token
embedding in NLP), a learned [class] token is prepended, and a learned
position embedding is added. The resulting token sequence is fed through
a standard Transformer encoder.

Eq. 1 (patch + position embedding):

    z_0 = [x_class ; x_p^1 E ; x_p^2 E ; ... ; x_p^N E] + E_pos

Eq. 2-3 (one encoder layer, applied L times):

    z'_l = MSA(LN(z_{l-1})) + z_{l-1}
    z_l  = MLP(LN(z'_l))    + z'_l

Eq. 4 (classification head input):

    y = LN(z_L^0)

Symbol -> code variable -> shape -> meaning:
    image                    images              [B,C,H,W]    input image batch
    P (patch size)           patch_size          scalar       side length of one square patch
    x_p^i (flattened patch)  patches             [B,N,P*P*C]  N patches, each flattened to P*P*C values
    N (#patches)             num_patches         scalar       (H/P) * (W/P), the token count contributed by the image
    E (patch projection)     patch_embedding.projection.weight [D, P*P*C]  learned linear projection, paper's E
    x_p^i E                  patch_tokens        [B,N,D]      projected patch tokens
    x_class                  cls_token           [1,1,D]      learned, broadcast to every batch element
    z_0 (before encoder)     x                    [B,N+1,D]    cls_token concatenated in front of patch_tokens, + position_embedding
    E_pos                    position_embedding  [1,N+1,D]    learned, one vector per position (0 = CLS)
    encoder stack output     x (after blocks)    [B,N+1,D]    shape-preserving stack of TransformerEncoderBlock
    y (classifier input)     cls_output          [B,D]        x[:,0,:], the position-0 (CLS) token after all blocks
    classifier weights       classifier_head     [D,num_classes]
    logits                   logits              [B,num_classes]

Indexing convention: position 0 along the token axis is always the CLS
token; positions 1..N are patch tokens in row-major order over the
(H/P, W/P) patch grid (patch (0,0), then (0,1), ..., row by row) --
this fixed ordering is what lets position_embedding be a plain learned
[1,N+1,D] tensor added by broadcasting rather than looked up, and is
also what makes the position-embedding-similarity visualization below
interpretable as a 2D grid.

Eq. 2-3 are Pre-LN (LayerNorm applied *before* each sub-layer, then a
residual add) -- the opposite wrapping from the original Vaswani et al.,
2017 Transformer's Post-LN (LayerNorm *after* the residual add), which
code/transformers/transformer_block.py implements for that (separate)
exercise. ViTEncoderBlockPreLN below implements Eq. 2-3 exactly, with
its own MultiHeadSelfAttention and PositionwiseFeedForward. These two
sub-modules are intentionally duplicated from, not imported from,
transformer_block.py: this file is kept fully independent so edits to
the Vaswani-Transformer exercise can never change ViT's behavior (and
vice versa). The attention/FFN math itself is identical either way --
only where LayerNorm sits relative to the residual add differs.

Local usage (lightweight, CPU, no dataset, no GPU):
    python code/transformers/vit_patch_embedding.py --smoke-test
"""

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def manual_softmax(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """softmax(scores)_j = exp(scores_j) / sum_j' exp(scores_j'), with the
    standard max-subtraction trick for numerical stability. Duplicated
    from self_attention_qkv.py (see that file for the full derivation)
    rather than imported, to keep this file independent."""
    scores_shifted = scores - scores.amax(dim=dim, keepdim=True)
    exp_scores = torch.exp(scores_shifted)
    return exp_scores / exp_scores.sum(dim=dim, keepdim=True)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention, Sec. 3.2.2 of Vaswani et al., 2017
    (the attention mechanism itself, not the Pre/Post-LN wrapping around
    it). Duplicated from transformer_block.py rather than imported --
    see the module docstring above for why."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads (d_k = embed_dim / num_heads)"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        """[B,N,D] -> [B,h,N,d_k]."""
        batch_size, num_tokens, embed_dim = t.shape
        t = t.view(batch_size, num_tokens, self.num_heads, self.head_dim)  # [B,N,D] -> [B,N,h,d_k]
        return t.transpose(1, 2)  # [B,N,h,d_k] -> [B,h,N,d_k]

    def _merge_heads(self, t: torch.Tensor) -> torch.Tensor:
        """[B,h,N,d_k] -> [B,N,D], inverse of _split_heads."""
        batch_size, num_heads, num_tokens, head_dim = t.shape
        t = t.transpose(1, 2)  # [B,h,N,d_k] -> [B,N,h,d_k]
        return t.contiguous().view(batch_size, num_tokens, num_heads * head_dim)  # [B,N,h,d_k] -> [B,N,D]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: [B,N,D]
        Returns:
            attention_weights: [B,h,N,N]
            output:            [B,N,D]
        """
        batch_size, num_tokens, embed_dim = x.shape
        assert embed_dim == self.embed_dim

        q = self._split_heads(self.W_q(x))  # [B,h,N,d_k]
        k = self._split_heads(self.W_k(x))  # [B,h,N,d_k]
        v = self._split_heads(self.W_v(x))  # [B,h,N,d_k]

        raw_scores = torch.matmul(q, k.transpose(-2, -1))  # [B,h,N,d_k] @ [B,h,d_k,N] -> [B,h,N,N]
        scaled_scores = raw_scores / math.sqrt(self.head_dim)
        attention_weights = manual_softmax(scaled_scores, dim=-1)  # [B,h,N,N]

        head_outputs = torch.matmul(attention_weights, v)  # [B,h,N,N] @ [B,h,N,d_k] -> [B,h,N,d_k]
        context = self._merge_heads(head_outputs)  # [B,h,N,d_k] -> [B,N,D]
        output = self.W_o(context)  # [B,N,D]

        return {"attention_weights": attention_weights, "output": output}


class PositionwiseFeedForward(nn.Module):
    """FFN(y) = max(0, y W_1 + b_1) W_2 + b_2, Sec. 3.3. Duplicated from
    transformer_block.py rather than imported -- see module docstring."""

    def __init__(self, embed_dim: int, ffn_hidden_dim: int):
        super().__init__()
        self.ffn_linear1 = nn.Linear(embed_dim, ffn_hidden_dim)
        self.relu = nn.ReLU()
        self.ffn_linear2 = nn.Linear(ffn_hidden_dim, embed_dim)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """y: [B,N,D] -> [B,N,D]."""
        return self.ffn_linear2(self.relu(self.ffn_linear1(y)))


class ViTEncoderBlockPreLN(nn.Module):
    """One ViT encoder layer, Eq. 2-3 of Dosovitskiy et al., 2021 (Pre-LN:
    LayerNorm applied *before* each sub-layer, residual added around the
    whole LN(sub-layer(.)) term):

        z'_l = MSA(LN(z_{l-1})) + z_{l-1}
        z_l  = MLP(LN(z'_l))    + z'_l

    This is the opposite wrapping from transformer_block.py's Post-LN
    TransformerEncoderBlock (LayerNorm(x + Sublayer(x))); see the module
    docstring for why both exist independently."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_hidden_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = PositionwiseFeedForward(embed_dim, ffn_hidden_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: [B,N,D] (z_{l-1} in Eq. 2-3)
        Returns:
            attn_attention_weights: [B,h,N,N]
            z:                      [B,N,D] (z_l in Eq. 2-3), block output
        """
        attn_intermediates = self.self_attention(self.norm1(x))  # MSA(LN(z_{l-1}))
        z_prime = x + attn_intermediates["output"]  # [B,N,D], Eq. 2: z'_l = MSA(LN(z_{l-1})) + z_{l-1}

        ffn_out = self.feed_forward(self.norm2(z_prime))  # MLP(LN(z'_l))
        z = z_prime + ffn_out  # [B,N,D], Eq. 3: z_l = MLP(LN(z'_l)) + z'_l

        return {"attn_attention_weights": attn_intermediates["attention_weights"], "z": z}


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Splits a batch of images into flattened, non-overlapping square
    patches, in row-major order over the patch grid.

    Args:
        images:     [B,C,H,W], H and W must both be divisible by patch_size
        patch_size: P, side length of one square patch

    Returns:
        patches: [B,N,P*P*C], N = (H/P)*(W/P)
    """
    batch_size, num_channels, height, width = images.shape
    assert height % patch_size == 0 and width % patch_size == 0, (
        f"image size ({height},{width}) must be divisible by patch_size {patch_size}"
    )
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size

    # unfold(dim=2, P, P): slides a size-P window over H with stride P,
    # replacing axis H with two axes (num_patches_h, P): [B,C,H,W] -> [B,C,num_patches_h,W,P]
    patches = images.unfold(2, patch_size, patch_size)
    # unfold(dim=3, P, P): same trick over W (now at axis 3): [B,C,num_patches_h,W,P] -> [B,C,num_patches_h,num_patches_w,P,P]
    patches = patches.unfold(3, patch_size, patch_size)

    # Move the two patch-grid axes next to the batch axis, keep (C,P,P) as
    # the trailing "flatten into one patch vector" axes, in that order --
    # this exact order (C,P,P) is what makes the Conv2d-equivalence check
    # in _smoke_test line up with nn.Conv2d's [out,in,kH,kW] weight layout.
    patches = patches.permute(0, 2, 3, 1, 4, 5)  # [B,C,nh,nw,P,P] -> [B,nh,nw,C,P,P]

    # .contiguous() is required before .view(): permute() returns a
    # non-contiguous view, and .view() refuses to silently copy memory.
    patches = patches.contiguous().view(
        batch_size, num_patches_h * num_patches_w, num_channels * patch_size * patch_size
    )  # [B,nh,nw,C,P,P] -> [B,N,C*P*P]
    return patches


class PatchEmbedding(nn.Module):
    """Patch + linear projection, Eq. 1's "x_p^i E" term: flattens each
    image patch and linearly projects it to embed_dim (paper's E matrix).
    The CLS token and position embedding (the rest of Eq. 1) are added in
    VisionTransformerClassifier below, since they belong to the whole
    token sequence, not to an individual patch."""

    def __init__(self, image_channels: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.patch_dim = image_channels * patch_size * patch_size  # P*P*C
        self.projection = nn.Linear(self.patch_dim, embed_dim)  # E: [patch_dim, D] (nn.Linear weight: [D, patch_dim])

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            images: [B,C,H,W]

        Returns:
            patches:     [B,N,patch_dim], flattened patches before projection
            patch_tokens: [B,N,D], patches @ E
        """
        patches = patchify(images, self.patch_size)  # [B,C,H,W] -> [B,N,patch_dim]
        patch_tokens = self.projection(patches)  # [B,N,patch_dim] -> [B,N,D]
        return {"patches": patches, "patch_tokens": patch_tokens}


class VisionTransformerClassifier(nn.Module):
    """Minimal ViT: PatchEmbedding -> [CLS; patch tokens] + position
    embedding -> N x ViTEncoderBlockPreLN (Eq. 2-3) -> Linear(CLS)."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        image_channels: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ffn_hidden_dim: int,
        num_classes: int,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches_per_side = image_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2  # N
        self.embed_dim = embed_dim

        self.patch_embedding = PatchEmbedding(image_channels, patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)  # [1,1,D]
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)  # [1,N+1,D]

        self.blocks = nn.ModuleList(
            [ViTEncoderBlockPreLN(embed_dim, num_heads, ffn_hidden_dim) for _ in range(num_layers)]
        )
        self.classifier_head = nn.Linear(embed_dim, num_classes)  # [D,num_classes]

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            images: [B,C,H,W], C == image_channels, H == W == image_size

        Returns:
            patches:      [B,N,patch_dim]
            patch_tokens: [B,N,D]
            x:            [B,N+1,D], final-layer token representations (CLS + patches)
            attn_weights: list of length num_layers, each [B,heads,N+1,N+1]
            cls_output:   [B,D], final-layer CLS representation
            logits:       [B,num_classes]
        """
        batch_size = images.shape[0]

        patch_out = self.patch_embedding(images)
        patch_tokens = patch_out["patch_tokens"]  # [B,N,D]

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [1,1,D] -> [B,1,D]
        x = torch.cat([cls_tokens, patch_tokens], dim=1)  # [B,1,D]+[B,N,D] -> [B,N+1,D], CLS prepended at position 0

        x = x + self.position_embedding  # [B,N+1,D] + [1,N+1,D] -> [B,N+1,D], broadcast over batch

        attn_weights = []
        for block in self.blocks:
            intermediates = block(x)
            x = intermediates["z"]  # [B,N+1,D], shape-preserving
            attn_weights.append(intermediates["attn_attention_weights"])  # [B,heads,N+1,N+1]

        cls_output = x[:, 0, :]  # [B,N+1,D] -> [B,D], CLS = position 0 after all blocks
        logits = self.classifier_head(cls_output)  # [B,D] -> [B,num_classes]

        return {
            "patches": patch_out["patches"],
            "patch_tokens": patch_tokens,
            "x": x,
            "attn_weights": attn_weights,
            "cls_output": cls_output,
            "logits": logits,
        }


def print_vit_shapes(images: torch.Tensor, outputs: dict, model: VisionTransformerClassifier) -> None:
    """Prints the shape at every major stage, with the token-count
    bookkeeping (N = num_patches, N+1 with CLS) made explicit."""
    print(f"images (input)                          : {tuple(images.shape)}")
    print(f"patch_size                               : {model.patch_size}")
    print(f"num_patches N = (H/P)*(W/P)               : {model.num_patches} "
          f"({model.num_patches_per_side} x {model.num_patches_per_side})")
    print(f"embed_dim D                              : {model.embed_dim}")
    print(f"patches (flattened, pre-projection)      : {tuple(outputs['patches'].shape)}")
    print(f"patch_tokens (= patches @ E)             : {tuple(outputs['patch_tokens'].shape)}")
    print(f"x (after CLS + position embedding)       : {tuple(outputs['x'].shape)}")
    print(f"attn_weights[-1] (last-layer attention)  : {tuple(outputs['attn_weights'][-1].shape)}")
    print(f"cls_output (= x[:,0,:])                  : {tuple(outputs['cls_output'].shape)}")
    print(f"logits                                   : {tuple(outputs['logits'].shape)}")


def visualize_patch_grid(image: torch.Tensor, patch_size: int, save_path: str | None = None):
    """Draws the patch grid lines over one image, so the "token count"
    concept (N patches = N tokens fed to the encoder) is visible, not
    just printed as a number.

    Args:
        image:      [C,H,W], values assumed already in a displayable
                    range (e.g. [0,1]); channel-first, converted to
                    channel-last for imshow
        patch_size: P
        save_path:  if given, save instead of showing interactively
    """
    num_channels, height, width = image.shape
    image_for_display = image.detach().permute(1, 2, 0).numpy()  # [C,H,W] -> [H,W,C] for imshow
    if num_channels == 1:
        image_for_display = image_for_display[:, :, 0]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image_for_display, cmap="gray" if num_channels == 1 else None)
    for grid_line in range(patch_size, width, patch_size):
        ax.axvline(grid_line - 0.5, color="red", linewidth=0.8)
    for grid_line in range(patch_size, height, patch_size):
        ax.axhline(grid_line - 0.5, color="red", linewidth=0.8)
    num_patches_per_side = height // patch_size
    ax.set_title(
        f"{num_patches_per_side}x{num_patches_per_side} = {num_patches_per_side ** 2} patches "
        f"of size {patch_size}x{patch_size}",
        fontsize=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        print(f"saved patch-grid figure to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def visualize_position_embedding_similarity(
    position_embedding: torch.Tensor,
    num_patches_per_side: int,
    reference_patch_indices: list[int],
    save_path: str | None = None,
):
    """Simplified version of the ViT paper's Fig. 7: for a handful of
    reference patches, plots the cosine similarity of that patch's
    learned position embedding to every other patch's, reshaped back
    into the (H/P, W/P) patch grid. If position embeddings have learned
    useful 2D structure, each reference patch should be most similar to
    patches spatially near it in the grid (a bright spot near that
    patch's own grid location). (Simplified from the paper: the paper
    plots this grid for every patch; here only a few reference patches
    are shown, which is enough to see the structure without 64+ subplots.)

    Args:
        position_embedding:      [1,N+1,D] (index 0 is CLS, indices 1..N are patches)
        num_patches_per_side:    H/P == W/P
        reference_patch_indices: 0-indexed patch indices (into the N patches, CLS excluded) to use as reference
        save_path:                if given, save instead of showing interactively
    """
    num_patches = num_patches_per_side ** 2
    patch_position_embedding = position_embedding[0, 1:, :]  # [1,N+1,D] -> [N,D], drop CLS (index 0)
    assert patch_position_embedding.shape[0] == num_patches

    # Cosine similarity between every pair of patch position embeddings: [N,D] -> [N,N].
    normalized = F.normalize(patch_position_embedding, dim=-1)  # [N,D], unit-norm rows
    similarity = normalized @ normalized.T  # [N,D] @ [D,N] -> [N,N]

    num_reference = len(reference_patch_indices)
    fig, axes = plt.subplots(1, num_reference, figsize=(3.2 * num_reference, 3.2))
    if num_reference == 1:
        axes = [axes]
    for ax, reference_index in zip(axes, reference_patch_indices):
        similarity_row = similarity[reference_index]  # [N], similarity of this patch to every patch
        similarity_grid = similarity_row.detach().reshape(num_patches_per_side, num_patches_per_side).numpy()  # [N] -> [H/P,W/P]
        im = ax.imshow(similarity_grid, cmap="viridis")
        reference_row, reference_col = divmod(reference_index, num_patches_per_side)
        ax.scatter([reference_col], [reference_row], marker="x", color="red", s=60)  # mark the reference patch itself
        ax.set_title(f"patch {reference_index}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("position-embedding cosine similarity (red x = reference patch)", fontsize=10)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)

    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        print(f"saved position-embedding-similarity figure to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _smoke_test():
    """Lightweight local checks: patchify shapes, patchify+linear exactly
    equals an equivalent nn.Conv2d (mathematical equation-equivalence
    check, not merely "looks plausible"), ViTEncoderBlockPreLN's forward()
    exactly matches an independently-reconstructed Eq. 2-3 computation,
    full-model shapes end-to-end, per-head attention rows sum to 1,
    finite values, gradients flow to every parameter, and fixed-seed
    reproducibility. No dataset, no GPU."""
    torch.manual_seed(0)

    # --- ViTEncoderBlockPreLN forward() vs. a hand-reconstructed Eq. 2-3 ---
    embed_dim, num_heads, ffn_hidden_dim, num_tokens = 8, 2, 16, 5
    block = ViTEncoderBlockPreLN(embed_dim, num_heads, ffn_hidden_dim)
    x_block = torch.randn(3, num_tokens, embed_dim)

    attn_intermediates = block.self_attention(block.norm1(x_block))  # MSA(LN(z_{l-1}))
    z_prime_manual = x_block + attn_intermediates["output"]  # Eq. 2: z'_l = MSA(LN(z_{l-1})) + z_{l-1}
    ffn_out_manual = block.feed_forward(block.norm2(z_prime_manual))  # MLP(LN(z'_l))
    z_manual = z_prime_manual + ffn_out_manual  # Eq. 3: z_l = MLP(LN(z'_l)) + z'_l
    assert torch.allclose(z_manual, block(x_block)["z"], atol=1e-6), "forward() must exactly match Eq. 2-3"

    # Pre-LN must genuinely differ from Post-LN (LayerNorm(x + Sublayer(x)))
    # for the same submodules -- confirms the wrapping, not just the
    # sub-layer math, is what's being tested above.
    post_ln_z = block.norm1(x_block + block.self_attention(x_block)["output"])
    assert not torch.allclose(z_prime_manual, post_ln_z, atol=1e-3), "Pre-LN and Post-LN should not coincide"

    # --- patchify + PatchEmbedding vs. equivalent nn.Conv2d ---
    batch_size, num_channels, image_size, patch_size, embed_dim = 2, 3, 8, 4, 6
    images = torch.randn(batch_size, num_channels, image_size, image_size)

    patches = patchify(images, patch_size)
    num_patches_per_side = image_size // patch_size
    num_patches = num_patches_per_side ** 2
    patch_dim = num_channels * patch_size * patch_size
    assert patches.shape == (batch_size, num_patches, patch_dim)

    patch_embedding = PatchEmbedding(num_channels, patch_size, embed_dim)
    patch_tokens = patch_embedding(images)["patch_tokens"]
    assert patch_tokens.shape == (batch_size, num_patches, embed_dim)

    # Equation-equivalence check: patchify(images) @ E (the "manual" path
    # above) must exactly equal an nn.Conv2d with kernel_size=stride=P
    # using the *same* weight, reshaped from [D,patch_dim] -> [D,C,P,P] --
    # this is the standard efficient implementation trick, and reshaping
    # the linear weight that way is valid only because patchify() flattens
    # each patch in (C,P,P) order, matching Conv2d's weight layout exactly.
    conv_weight = patch_embedding.projection.weight.view(embed_dim, num_channels, patch_size, patch_size)
    conv_bias = patch_embedding.projection.bias
    conv_output = F.conv2d(images, conv_weight, conv_bias, stride=patch_size)  # [B,D,nh,nw]
    conv_tokens = conv_output.flatten(2).transpose(1, 2)  # [B,D,nh,nw] -> [B,D,N] -> [B,N,D]
    assert torch.allclose(patch_tokens, conv_tokens, atol=1e-5), "patchify+linear must exactly equal equivalent Conv2d"

    # --- full VisionTransformerClassifier ---
    num_heads, num_layers, ffn_hidden_dim, num_classes = 2, 2, 16, 5
    model = VisionTransformerClassifier(
        image_size, patch_size, num_channels, embed_dim, num_heads, num_layers, ffn_hidden_dim, num_classes
    )
    images_grad = images.clone().requires_grad_(True)
    outputs = model(images_grad)

    assert outputs["patch_tokens"].shape == (batch_size, num_patches, embed_dim)
    assert outputs["x"].shape == (batch_size, num_patches + 1, embed_dim)
    assert len(outputs["attn_weights"]) == num_layers
    assert outputs["attn_weights"][-1].shape == (batch_size, num_heads, num_patches + 1, num_patches + 1)
    assert outputs["cls_output"].shape == (batch_size, embed_dim)
    assert outputs["logits"].shape == (batch_size, num_classes)
    assert torch.isfinite(outputs["logits"]).all()

    row_sums = outputs["attn_weights"][-1].sum(dim=-1)  # [B,heads,N+1]
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    loss = outputs["logits"].sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} gradient is not finite"

    with torch.no_grad():
        logits_repeat = model(images)["logits"]
    assert torch.allclose(outputs["logits"], logits_repeat, atol=1e-5)

    print("vit_patch_embedding smoke test passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="run lightweight local checks and exit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ffn-hidden-dim", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=10)
    args = parser.parse_args()

    if args.smoke_test:
        _smoke_test()
    else:
        torch.manual_seed(args.seed)
        image_channels = 3
        images = torch.rand(1, image_channels, args.image_size, args.image_size)  # demo image, no dataset needed here

        model = VisionTransformerClassifier(
            args.image_size, args.patch_size, image_channels, args.embed_dim,
            args.num_heads, args.num_layers, args.ffn_hidden_dim, args.num_classes,
        )
        outputs = model(images)
        print_vit_shapes(images, outputs, model)

        visualize_patch_grid(images[0], args.patch_size)
        mid_patch = (model.num_patches_per_side // 2) * model.num_patches_per_side + model.num_patches_per_side // 2
        visualize_position_embedding_similarity(
            model.position_embedding, model.num_patches_per_side, reference_patch_indices=[0, mid_patch]
        )
