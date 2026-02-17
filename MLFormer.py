import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
import torch.utils.checkpoint as checkpoint
from einops import rearrange
import torchprofile
from thop import profile
from timm.models.layers import to_2tuple

class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)

class Scale_reduce(nn.Module):
    def __init__(self, dim, reduction_ratio):
        super().__init__()
        self.dim = dim
        self.reduction_ratio = reduction_ratio
        if(len(self.reduction_ratio)==4):
            self.sr0 = nn.Conv2d(dim, dim, reduction_ratio[3], reduction_ratio[3])
            self.sr1 = nn.Conv2d(dim*2, dim*2, reduction_ratio[2], reduction_ratio[2])
            self.sr2 = nn.Conv2d(dim*5, dim*5, reduction_ratio[1], reduction_ratio[1])

        elif(len(self.reduction_ratio)==3):
            self.sr0 = nn.Conv2d(dim*2, dim*2, reduction_ratio[2], reduction_ratio[2])
            self.sr1 = nn.Conv2d(dim*5, dim*5, reduction_ratio[1], reduction_ratio[1])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        if(len(self.reduction_ratio)==4):
            tem0 = x[:,:3136,:].reshape(B, 56, 56, C).permute(0, 3, 1, 2)
            tem1 = x[:,3136:4704,:].reshape(B, 28, 28, C*2).permute(0, 3, 1, 2)
            tem2 = x[:,4704:5684,:].reshape(B, 14, 14, C*5).permute(0, 3, 1, 2)
            tem3 = x[:,5684:6076,:]

            sr_0 = self.sr0(tem0).reshape(B, C, -1).permute(0, 2, 1)
            sr_1 = self.sr1(tem1).reshape(B, C, -1).permute(0, 2, 1)
            sr_2 = self.sr2(tem2).reshape(B, C, -1).permute(0, 2, 1)

            reduce_out = self.norm(torch.cat([sr_0, sr_1, sr_2, tem3], -2))

        if(len(self.reduction_ratio)==3):
            tem0 = x[:,:1568,:].reshape(B, 28, 28, C*2).permute(0, 3, 1, 2)
            tem1 = x[:,1568:2548,:].reshape(B, 14, 14, C*5).permute(0, 3, 1, 2)
            tem2 = x[:,2548:2940,:]

            sr_0 = self.sr0(tem0).reshape(B, C, -1).permute(0, 2, 1)
            sr_1 = self.sr1(tem1).reshape(B, C, -1).permute(0, 2, 1)

            reduce_out = self.norm(torch.cat([sr_0, sr_1, tem2], -2))

        return reduce_out

class M_EfficientSelfAtten(nn.Module):
    def __init__(self, dim, head, reduction_ratio):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio # list[1  2  4  8]
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim*2, bias=True)
        self.proj = nn.Linear(dim, dim)

        if reduction_ratio is not None:
            self.scale_reduce = Scale_reduce(dim,reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.reduction_ratio is not None:
            x = self.scale_reduce(x)

        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)

        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)


        return out

class RoPE(torch.nn.Module):
    r"""Rotary Positional Embedding.
    """
    def __init__(self, shape, base=10000):
        super(RoPE, self).__init__()

        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % k_max == 0

        # angles
        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)

        # rotation
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        self.register_buffer('rotations', rotations)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(self.rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)

class LinearAttention(nn.Module):
    r""" Linear Attention with LePE and RoPE.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, input_resolution, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE(shape=(input_resolution[0], input_resolution[1], dim))

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, N, C)
        """
        b, n, c = x.shape  # 例：4,3136,64
        h = int(n ** 0.5)
        w = int(n ** 0.5)
        num_heads = self.num_heads
        head_dim = c // num_heads
        # 例：4,3136,64 --> 4,3136,128 --> 4,3136,2,64 --> 2,4,3136,64
        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)

        q, k, v = qk[0], qk[1], x
        # q, k, v: b, n, c

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        q_rope = self.rope(q.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q_rope @ kv * z

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.lepe(v).permute(0, 2, 3, 1).reshape(b, n, c)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'

class Stem(nn.Module):
    r""" Stem

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size) # 224 224
        patch_size = to_2tuple(patch_size) # 4 4
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]] # 56 56
        self.img_size = img_size # 224
        self.patch_size = patch_size # 4
        self.patches_resolution = patches_resolution # 56,56
        self.num_patches = patches_resolution[0] * patches_resolution[1] # 3136

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.conv1 = ConvLayer(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False, act_func=None)
        )
        self.conv3 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim * 4, kernel_size=3, stride=2, padding=1, bias=False),
            ConvLayer(embed_dim * 4, embed_dim, kernel_size=1, bias=False, act_func=None)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.conv1(x) # 4,32,112,112
        x = self.conv2(x) + x # 4,32,112,112
        x = self.conv3(x) # 4,64,56,56
        x = x.flatten(2).transpose(1, 2)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MixFFN_skip(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.norm1 = nn.LayerNorm(hidden_features)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W)+self.fc1(x)))
        out = self.fc2(ax)
        return out




class MLLABlock(nn.Module):
    r""" MLLA Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.attn = LinearAttention(dim=dim, input_resolution=input_resolution, num_heads=num_heads, qkv_bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.mix_FFN =MixFFN_skip(dim,dim)
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x + self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        shortcut = x

        x = self.norm1(x)
        act_res = self.act(self.act_proj(x))
        x = self.in_proj(x).view(B, H, W, C)
        x = self.act(self.dwc(x.permute(0, 3, 1, 2))).permute(0, 2, 3, 1).view(B, L, C)

        # Linear Attention
        x = self.attn(x)

        x = self.out_proj(x * act_res)
        x = shortcut + self.drop_path(x)
        x = x + self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        # FFN 1 MLLAUNET中FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # FFN2 根据missformer改版FFN
        # x = x + self.drop_path(self.mix_FFN(self.norm2(x),H,W))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"mlp_ratio={self.mlp_ratio}"


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, dropout=0, norm=nn.BatchNorm2d, act_func=nn.ReLU):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=bias,
        )
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act_func() if act_func else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x



class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
    """

    def __init__(self, input_resolution, dim, ratio=4.0):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        in_channels = dim
        out_channels = 2 * dim
        self.conv = nn.Sequential(
            ConvLayer(in_channels, int(out_channels * ratio), kernel_size=1, norm=None),
            ConvLayer(int(out_channels * ratio), int(out_channels * ratio), kernel_size=3, stride=2, padding=1, groups=int(out_channels * ratio), norm=None),
            ConvLayer(int(out_channels * ratio), out_channels, kernel_size=1, act_func=None)
        )

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = self.conv(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        return x


class BasicLayer(nn.Module):
    """ A basic MLLA layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.norm = norm_layer(dim)
        # build blocks
        self.blocks = nn.ModuleList([
            MLLABlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim)
        else:
            self.downsample = None

    def forward(self, x):
        B = x.shape[0]
        H,W = self.input_resolution
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            tx = self.norm(x).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            x = self.downsample(x)
            return x,tx
        tx =self.norm(x).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x,tx

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class BasicLayer_up(nn.Module):
    """ A basic MLLA layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        upsample (nn.Module | None, optional): Upsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            MLLABlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            for i in range(depth)])

        # patch expanding layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, ratio=4.0, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"



class BridgeLayer_4(nn.Module):
    def __init__(self, dims, head, reduction_ratios):
        super().__init__()

        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten(dims, head, reduction_ratios)
        self.norm2 = nn.LayerNorm(dims)
        self.mixffn1 = MixFFN_skip(dims, dims * 4)
        self.mixffn2 = MixFFN_skip(dims * 2, dims * 8)
        self.mixffn3 = MixFFN_skip(dims * 4, dims * 16)
        self.mixffn4 = MixFFN_skip(dims * 8, dims * 32)

    def forward(self, inputs):
        B = inputs[0].shape[0] # 4
        C = 64
        if (type(inputs) == list):
            c1, c2, c3, c4 = inputs
            B, C, _, _ = c1.shape
            c1f = c1.permute(0, 2, 3, 1).reshape(B, -1, C)  # 3136*64
            c2f = c2.permute(0, 2, 3, 1).reshape(B, -1, C)  # 1568*64
            c3f = c3.permute(0, 2, 3, 1).reshape(B, -1, C)  # 980*64   /// 784(若输入为4,256,14,14）
            c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64

            # print(c1f.shape, c2f.shape, c3f.shape, c4f.shape)
            inputs = torch.cat([c1f, c2f, c3f, c4f], -2) # 4,6076,64
        else:
            B, _, C = inputs.shape

        tx1 = inputs + self.attn(self.norm1(inputs))
        tx = self.norm2(tx1)

        tem1 = tx[:, :3136, :].reshape(B, -1, C)
        tem2 = tx[:, 3136:4704, :].reshape(B, -1, C * 2)
        tem3 = tx[:, 4704:5488, :].reshape(B, -1, C * 4)
        tem4 = tx[:, 5488:5880, :].reshape(B, -1, C * 8)

        m1f = self.mixffn1(tem1, 56, 56).reshape(B, -1, C)
        m2f = self.mixffn2(tem2, 28, 28).reshape(B, -1, C)
        m3f = self.mixffn3(tem3, 14, 14).reshape(B, -1, C)
        m4f = self.mixffn4(tem4, 7, 7).reshape(B, -1, C)

        t1 = torch.cat([m1f, m2f, m3f, m4f], -2)

        tx2 = tx1 + t1

        return tx2


class BridegeBlock_4(nn.Module):
    def __init__(self, dims, head, reduction_ratios):
        super().__init__()
        self.bridge_layer1 = BridgeLayer_4(dims, head, reduction_ratios)
        self.bridge_layer2 = BridgeLayer_4(dims, head, reduction_ratios)
        self.bridge_layer3 = BridgeLayer_4(dims, head, reduction_ratios)
        self.bridge_layer4 = BridgeLayer_4(dims, head, reduction_ratios)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape
        # torch.Size([4, 64, 56, 56])
        # torch.Size([4, 128, 28, 28])
        # torch.Size([4, 256, 14, 14])
        # torch.Size([4, 512, 7, 7])
        bridge1 = self.bridge_layer1(x)
        bridge2 = self.bridge_layer2(bridge1)
        bridge3 = self.bridge_layer3(bridge2)
        bridge4 = self.bridge_layer4(bridge3)
        B,_,C = bridge4.shape
        outs = []

        sk1 = bridge4[:,:3136,:].reshape(B, 56, 56, C).permute(0,3,1,2)
        sk2 = bridge4[:,3136:4704,:].reshape(B, 28, 28, C*2).permute(0,3,1,2)
        sk3 = bridge4[:,4704:5488,:].reshape(B, 14, 14, C*4).permute(0,3,1,2)
        sk4 = bridge4[:,5488:5880,:].reshape(B, 7, 7, C*8).permute(0,3,1,2)

        outs.append(sk1)
        outs.append(sk2)
        outs.append(sk3)
        outs.append(sk4)

        return outs

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, ratio=4.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        in_channels = dim
        out_channels = dim // 2  # 因为我们要扩展空间维度，所以通道数减半
        self.norm = norm_layer(out_channels)
        self.conv = nn.Sequential(
            ConvLayer(in_channels, int(in_channels * ratio), kernel_size=1, norm=None),
            nn.ConvTranspose2d(int(in_channels * ratio), int(in_channels * ratio), kernel_size=3, stride=2, padding=1,
                               output_padding=1, groups=int(in_channels * ratio), bias=False),
            ConvLayer(int(in_channels * ratio), out_channels, kernel_size=1, act_func=None)
        )

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # B, C, H, W
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).reshape(B, -1, self.dim // 2)
        x = self.norm(x)
        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x= self.norm(x.clone())

        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1,):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)
# depths=[2, 4, 8, 4],num_heads=[2, 4, 8, 16], depths_decoder=[1, 8, 4, 2]
class MLAFormer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=64, num_classes=9, depths=[2, 2, 2, 2],
                 num_heads=[2, 4, 8, 16], depths_decoder=[1, 2, 2, 2], mlp_ratio=4., qkv_bias=True, drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, ape=False, use_checkpoint=False, encoder_pretrained=True,
                 final_upsample="expand_first", **kwargs):
        '''
        img_size (int): 输入图片大小
        patch_size(int)： 每个patch大小
        in_chans ： 输入通道
        embed_dim ：经过patchembeding之后的通道数
        num_classes：分类数
        depths： 层数 list 表示每个层几个块
        num_heads： 注意力头

        '''
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)  # 4
        self.embed_dim = embed_dim
        self.ape = ape
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))  # 512
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        # patch_embeding
        self.patch_embed = Stem(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches # (224/4) * (224/4) = 3136
        patches_resolution = self.patch_embed.patches_resolution  # [(224/4),(224/4)]
        self.patches_resolution = patches_resolution
        # 绝对位置编码
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        # dropout
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 生成一个 Drop Path 概率列表（dpr），使用 torch.linspace 生成从 0 到 drop_path_rate 之间的线性间隔值，用于控制每个编码器层的 Drop Path 丢弃概率。
        # 如： depth一共有2+4+8+4=18 ，则dpr从0-0.2 之间随机生成符合正则化的18个数，[0,0.125.....,0.2]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        # ------------------------------------Encoder--------------------------------------------
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        #  -------------------------------------------------------------------------------------

        #------------------------------------bridge---------------------------------------------
        self.reduction_ratios = [1, 2, 4, 8]  # [1, 1, 1, 1]  [1, 2, 2, 4]  [2, 2, 4, 4]  [1, 2, 4, 8]
        self.bridge = BridegeBlock_4(dims=64, head=1, reduction_ratios=self.reduction_ratios)
        #----------------------------------------------------------------------------------------


        #-----------------------------------Decoder----------------------------------------------
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), ratio=4.0, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=(
                                         patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                         patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                         depth=depths_decoder[i_layer],
                                         num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=qkv_bias, drop=drop_rate,
                                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                             depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=norm_layer,
                                         upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        if self.final_upsample == "expand_first":
            self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
                                          dim_scale=4, dim=embed_dim)
            self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

        #--------------------------------------------------------------------------------------------------------------

        #----------------------------------------UP_X4-----------------------------------------------------------------
        if self.final_upsample == "expand_first":
            self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
                                          dim_scale=4, dim=embed_dim)
            # self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)


        #--------------------------------------segmentationHead--------------------------------------------------------
        self.segmentation_head = SegmentationHead(in_channels=embed_dim, out_channels=num_classes, kernel_size=3, )

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        for layer in self.layers:
            x,tx = layer(x)
            x_downsample.append(tx)


        x = self.norm(x)
        return x, x_downsample

    def forward_up_features(self, bridge):
        b, c, _, _ = bridge[3].shape  # 4,512,7,7
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(bridge[3].permute(0, 2, 3, 1).view(b, -1, c))  # 4,196,256
            else:
                x = torch.cat([x, bridge[3 - inx].permute(0, 2, 3, 1).view(b, -1, c // 2 ** inx)], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
        x = self.norm_up(x)
        return x

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W

        return x

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        # encoder
        x, x_downsample = self.forward_features(x) # 9.49 GFLOPs    11.3M
        # bridge
        bridge = self.bridge(x_downsample)  # 12 GFLOPs  15.2 M
        # decoder
        x = self.forward_up_features(bridge) # 6 GFLOPs 5M
        # up_x4
        x = self.up_x4(x)  # 1 GFLOPs
        # segmentation Head
        out  = self.segmentation_head(x)
        return out

if __name__ == '__main__':
    input_tensor = torch.rand(1, 3, 224, 224)  # Batch size 1, 3 channels, 64x64 image
    model = MLAFormer(num_classes=9)
    output = model(input_tensor)
    print("Output shape:", output.shape)
    print('# generator parameters:', 1.0 * sum(param.numel() for param in model.parameters()) / 1000000)
    print('GFLOPs:', torchprofile.profile_macs(model,input_tensor) / 1e9)
