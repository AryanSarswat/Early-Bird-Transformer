import torch
import torch.nn as nn
from torchsummary import summary

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size):
        """
        Patch Embedding layer

        Args:
            in_channels (int): Number of input channels
            patch_size (int): Patch size
            emb_size (int): Embedding dimension of the output patch
        """
        super().__init__()
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class Embedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, emb_size, num_classes):
        """
        Complete Embedding which includes the patch embedding and the positional embedding

        Args:
            image_size (int): Size of the image
            patch_size (int): Size of the patch
            in_channels (int): Number of input channels
            emb_size (int): Embedding dimension of the output patch
            num_classes (int): Number of classes
        """
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positional_embedding = nn.Parameter(torch.randn((image_size // patch_size) ** 2 + 1, emb_size))
        
    def forward(self, x):
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.positional_embedding
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        """
        Multi-head self attention module

        Args:
            dim (int): Dimension of the input
            heads (int): Number of heads
        """
        super().__init__()
        self.heads = heads
        assert dim % heads == 0, "Dimension must be divisible by number of heads"
        self.attn_dim = dim // heads
    
        self.qkv = nn.Linear(dim, dim*3)
        
        self.MHA = nn.MultiheadAttention(dim, heads, dropout=dropout)
        
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x, output_attentions=False):
        attn_output_weights = None
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        if output_attentions:
            attn_output, attn_output_weights = self.MHA(q, k, v, need_weights=True)
        else:
            attn_output = self.MHA(q, k, v)
        
        out = self.proj(attn_output)
        out = self.proj_drop(out)
        
        return out, attn_output_weights
        
class MLP(nn.Module):
    """
    A multi-layer perceptron module.
    """

    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.dense_1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.dense_2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x
            
class Block(nn.Module):
    """
    A single transformer block.
    """

    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.attention = Attention(dim, heads)
        self.layernorm_1 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim, dropout=dropout)
        self.layernorm_2 = nn.LayerNorm(dim)

    def forward(self, x, output_attentions=False):
        # Self-attention
        attention_output, attention_probs = self.attention(self.layernorm_1(x), output_attentions=output_attentions)
        # Skip connection
        x = x + attention_output
        # Feed-forward network
        mlp_output = self.mlp(self.layernorm_2(x))
        # Skip connection
        x = x + mlp_output
        # Return the transformer block's output and the attention probabilities (optional)
        return (x, attention_probs)
        
class Transformer(nn.Module):
    """
    The transformer encoder module.
    """

    def __init__(self, layers, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        # Create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        for _ in range(layers):
            block = Block(dim, heads, mlp_dim, dropout=dropout)
            self.blocks.append(block)

    def forward(self, x, output_attentions=False):
        # Calculate the transformer block's output for each block
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
        # Return the encoder's output and the attention probabilities (optional)
        return (x, all_attentions)
    
class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, emb_size, num_classes, depth, heads, mlp_dim, dropout=0.1):
        """
        Vision Transformer

        Args:
            image_size (int): Size of the image
            patch_size (int): Size of the patch
            in_channels (int): Number of input channels
            emb_size (int): Embedding dimension of the output patch
            num_classes (int): Number of classes
            depth (int): Number of transformer blocks
            heads (int): Number of attention heads
            mlp_dim (int): Dimension of the feedforward network
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.embedding = Embedding(image_size, patch_size, in_channels, emb_size, num_classes)
        self.transformer = Transformer(depth, emb_size, heads, mlp_dim, dropout)
        self.head = nn.Linear(emb_size, num_classes)
        
        self.apply(self.init_weights)
        
    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(Embedding):
            module.positional_embedding.data = nn.init.trunc_normal_(module.positional_embedding.data, mean=0.0, std=0.02)
            module.cls_token.data = nn.init.trunc_normal_(module.cls_token.data, mean=0.0, std=0.02)
            
if __name__ == "__main__":
    model = VisionTransformer(image_size=224, patch_size=16, in_channels=3, emb_size=768, num_classes=1000, depth=12, heads=12, mlp_dim=3072)
    summary(model, (3, 224, 224))