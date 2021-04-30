import math
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    r"""
    Implementation of patch embedding encoding layer.

    Args:
        image_size (int):
            Input image size.
        
        patch_size (int):
            Patch size, input image will be split into (image_size // patch_size) ^ 2 patches.

        in_channels (int):
            Input channel dimension (default = 3 for RGB).

        embed_dim (int):
            The embedding dimension.

    """
    def __init__(self, 
                 image_size=224, 
                 patch_size=16, 
                 in_channels=3, 
                 embed_dim=768):

        super(PatchEmbedding, self).__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class PositionalEmbedding(nn.Module):
    r"""
    Adds learnable positional embeddings to the inputs.

    """
    def __init__(self, seq_len, embed_dim):
        super(PositionalEmbedding, self).__init__()
        self.embedding = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        
    def forward(self, x):
        return x + self.embedding


class GELU(nn.Module):
    r"""
    Implementation of Gaussian Error Linerar Units activation function.
    See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`

    """
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) \
            * (x + 0.044715 * torch.pow(x, 3))))


class MLPBlock(nn.Module):
    r"""
    Implementation of the MLP / feed-forward block.

    Args:
        input_dim (int):
            Input dimension, same as the output dimension.
        
        hidden_dim (int):
            Dimension of the hidden fully-connected layer.

        dropout_rate (float):
            Probability of an element to be zeroed. Default: 0.5.

    """
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 dropout_rate=0.1):

        super(MLPBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.block(x)


class Attention(nn.Module):
    r"""
    Implementation of multi-head self-attention block.
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`

    Args:
        embed_dim (int):
            Embedding dimension.
        
        num_heads (int):
            Number of attention heads.

        atten_drop (float):
            Dropout rate of the attentive vector.

        proj_drop (float):
            Dropout rate of the output projection.

    """
    def __init__(self, 
                 embed_dim, 
                 num_heads, 
                 atten_drop=0.0, 
                 proj_drop=0.0):

        super(Attention, self).__init__()  

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim * 1)
        self.atten_drop = nn.Dropout(atten_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, l, c = x.size()
        assert c == self.embed_dim  
        
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(b, l, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0, ...], qkv[1, ...], qkv[2, ...]

        a = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        a = a.softmax(dim=-1)
        a = self.atten_drop(a)

        x = torch.matmul(a, v).transpose(1, 2).reshape(b, l, c)
        x = self.proj_drop(self.out_proj(x))

        return x


class EncoderBlock(nn.Module):
    r"""
    Implementation of the transformer encoder block.

    Args:
        embed_dim (int):
            Embedding dimension.

        num_heads (int):
            Number of attention heads.

        hidden_dim (int):
            Hidden dimension in `MLPBlock` module.

        atten_drop (float):
            Dropout rate of attentive vector in `Attention` module.

        proj_drop (float):
            Dropout rate of output projection in `Attention` module.

        mlp_drop (float):
            Dropout rate in `MLPBlock` module.

    """
    def __init__(self, 
                 embed_dim, 
                 num_heads, 
                 hidden_dim, 
                 atten_drop=0., 
                 proj_drop=0.):
                 
        super(EncoderBlock, self).__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.atten = Attention(embed_dim, num_heads, atten_drop, proj_drop)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(embed_dim, hidden_dim, proj_drop)

    def forward(self, x):
        x = x + self.atten(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    r"""
    Implementation of transformer encoder for feature extraction.
    See `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>.` 

    """
    def __init__(self, 
                 num_layers, 
                 embed_dim, 
                 num_heads, 
                 hidden_dim, 
                 seq_length,
                 atten_drop=0., 
                 proj_drop=0.):

        super(Transformer, self).__init__()

        self.pos_embedding = PositionalEmbedding(seq_length, embed_dim)
        self.dropout = nn.Dropout(proj_drop)
        self.norm = nn.LayerNorm(embed_dim)

        self.blocks = nn.ModuleList(
            [EncoderBlock(embed_dim, 
                          num_heads, 
                          hidden_dim, 
                          atten_drop, 
                          proj_drop) for _ in range(num_layers)])
        
        
    def forward(self, x):
        x = self.pos_embedding(x)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        return self.norm(x)


class PreLogitsLayer(nn.Module):
    r""" 
    Implementation of pre-logits layer.

    """
    def __init__(self, embed_dim, repr_dim=None):
        super(PreLogitsLayer, self).__init__()
        if repr_dim is not None:
            self.proj = nn.Sequential(
                nn.Linear(embed_dim, repr_dim),
                nn.Tanh()
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        return self.proj(x)


class ViT(nn.Module):
    def __init__(self,
                 image_size=384,
                 patch_size=16,
                 in_channels=3,
                 num_classes=1000,
                 embed_dim=768,
                 num_layers=12,
                 num_heads=12,
                 hidden_dim=3072,
                 atten_drop=0.,
                 proj_drop=0.1,
                 repr_dim=None,
                 **kwargs):

        super(ViT, self).__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim

        seq_length = (image_size // patch_size) ** 2 + 1

        # sub modules
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.transformer = Transformer(num_layers,
                                       embed_dim,
                                       num_heads,
                                       hidden_dim,
                                       seq_length,
                                       atten_drop,
                                       proj_drop,)

        self.pre_logits = PreLogitsLayer(embed_dim, repr_dim)
        
        repr_dim = repr_dim if repr_dim else embed_dim
        self.head = nn.Linear(repr_dim, num_classes)

    def forward(self, x):
        """
        Defined the  forward operation.
        """
        b, _, _, _ = x.size()
        x = self.patch_embedding(x)
        x = torch.cat([self.cls_token.expand(b, -1, -1), x], dim=1)

        x = self.transformer(x)
        x = self.pre_logits(x)

        # only support cls tocken now
        x = x[:, 0]

        return self.head(x)


def build_head(repr_dim, num_classes):
    num_classes = 1 if num_classes == 2 else num_classes
    return nn.Linear(repr_dim, num_classes)
