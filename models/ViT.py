import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels):
        """
        Patch Embedding module to convert images to tokens

        Args:
            embed_dim (int): _description_
            patch_size (int): _description_
            num_patches (int): _description_
            dropout (float): _description_
            in_channels (int): _description_
        """
        super().__init__()
        
        self.patchify = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=embed_dim,
                      kernel_size=patch_size,
                      stride=patch_size
                      ),
            nn.Flatten(2),
        )
        
        self.cls_token = nn.Parameter(torch.randn(size=(1, 1, embed_dim)), requires_grad=True)
        self.positional_embeddings = nn.Parameter(torch.randn(size=(1, num_patches + 1, embed_dim)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        positional_embeddings = self.positional_embeddings.expand(x.shape[0], -1, -1)
        
        x = self.patchify(x).permute(0, 2, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + positional_embeddings
        x = self.dropout(x)
        
        return x
    
class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, lrp=False):
        """
        Multi-head self attention module

        Args:
            dim (int): Dimension of the input
            heads (int): Number of heads
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "Dimension must be divisible by number of heads"
        self.head_dim = embed_dim // num_heads
        self.lrp = lrp
        
        self.qkv = nn.Linear(embed_dim, embed_dim*3)
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
        if self.lrp:
            self.head_weights = nn.Parameter(torch.randn(size=(1, 1, num_heads, self.head_dim))) # Batch, Tokens, Heads, Dims
        
    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        
        attn_output = F.scaled_dot_product_attention(q, k, v)
        attn_output = attn_output.permute(0, 2, 1, 3) # [Batch, Tokens, Head, Dims]
        
        if self.lrp:
            saliency_weights = self.head_weights.expand(batch_size, seq_length, -1, -1)
            attn_output = attn_output * saliency_weights
        
        attn_output = attn_output.reshape(batch_size, seq_length, self.embed_dim)
        
        out = self.proj(attn_output)
        out = self.proj_drop(out)
        
        return out
    
    def get_lrp_weights(self):
        if self.lrp:
            return self.head_weights
        else:
            return None
        
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

    def __init__(self, dim, heads, mlp_dim, dropout=0.1, lrp=False):
        super().__init__()
        self.attention = Attention(dim, heads, lrp=lrp)
        self.layernorm_1 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim, dropout=dropout)
        self.layernorm_2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Self-attention
        attention_output = self.attention(self.layernorm_1(x))
        attention_output = self.dropout(attention_output)        
        # Skip connection
        x = x + attention_output
        # Feed-forward network
        mlp_output = self.mlp(self.layernorm_2(x))
        # Skip connection
        x = x + mlp_output
        # Return the transformer block's output and the attention probabilities (optional)
        return x
    
    def get_lrp_weights(self):
        return self.attention.get_lrp_weights()        
        
class Transformer(nn.Module):
    """
    The transformer encoder module.
    """

    def __init__(self, layers, dim, heads, mlp_dim, dropout=0.1, lrp=False):
        super().__init__()
        # Create a list of transformer blocks
        self.lrp = lrp
        self.blocks = nn.ModuleList([])
        for _ in range(layers):
            block = Block(dim, heads, mlp_dim, dropout=dropout, lrp=lrp)
            self.blocks.append(block)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        # Return the encoder's output and the attention probabilities (optional)
        return x
    
    def get_all_lrp_weights(self):
        if self.lrp:
            return [block.get_lrp_weights() for block in self.blocks]
        else:
            return None
    
class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_patches, in_channels, embed_dim, num_classes, depth, heads, mlp_dim, device, dropout=0.1, lrp=False):
        """
        _summary_

        Args:
            image_size (_type_): _description_
            patch_size (_type_): _description_
            num_patches (_type_): _description_
            in_channels (_type_): _description_
            embed_dim (_type_): _description_
            num_classes (_type_): _description_
            depth (_type_): _description_
            heads (_type_): _description_
            mlp_dim (_type_): _description_
            device (_type_): _description_
            dropout (float, optional): _description_. Defaults to 0.1.
        """
        super().__init__()
        self.device = device
        self.embedding = PatchEmbedding(embed_dim, patch_size, num_patches, dropout, in_channels)
        self.transformer = Transformer(depth, embed_dim, heads, mlp_dim, dropout, lrp)
        #transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        #self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=depth)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )
        
        self.apply(self.init_weights)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)[:,0,:]
        #print(transformer_out.shape)
        x = self.mlp_head(x)
        
        return x
    
    def get_lrp_weights(self):
        return self.transformer.get_all_lrp_weights()
    
    def get_lrp_weights_sum(self):
        return sum(i.abs().sum() for i in self.transformer.get_all_lrp_weights())
    
    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, PatchEmbedding):
            module.positional_embeddings.data = nn.init.trunc_normal_(module.positional_embeddings.data, mean=0.0, std=0.02)
            module.cls_token.data = nn.init.trunc_normal_(module.cls_token.data, mean=0.0, std=0.02)
            
if __name__ == "__main__":
    NUM_CLASSES = 10
    PATCH_SIZE = 4
    IMG_SIZE = 32
    IN_CHANNELS = 3
    NUM_HEADS = 8
    DROPOUT = 0.1
    HIDDEN_DIM = 512
    EMBED_DIM = 128
    NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
    DEPTH = 8
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LRP = False
    
    model = VisionTransformer(image_size=IMG_SIZE, 
                              patch_size=PATCH_SIZE, 
                              num_patches=NUM_PATCHES, 
                              in_channels=IN_CHANNELS, 
                              embed_dim=EMBED_DIM, 
                              num_classes=NUM_CLASSES, 
                              depth=DEPTH, 
                              heads=NUM_HEADS, 
                              mlp_dim=HIDDEN_DIM, 
                              device=DEVICE,
                              dropout=DROPOUT,
                              lrp=LRP).to(DEVICE)
    
    test = torch.randn(512,3,32,32).to(DEVICE)
    out = model(test)
    print(out.shape)
    
    #print(model.get_lrp_weights_sum())
    
    summary(model, (3,32,32))
    
    
    
    