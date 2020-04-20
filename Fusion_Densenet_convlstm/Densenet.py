from torch import nn
from layers import DenseBlock, TransitionDown, Bottleneck, TransitionUp
from __future__ import print_function
import torch, torchvision

# 全卷积DenseNet
'''
args:
    in_channels: 输入的通道数
    down_blocks: encoder过程中每个denseblock的层数
    up_blocks:   decoder过程中每个denseblock的层数
    bottleneck_layers: encoder与decoder中间一个denseblock的层数
    growth_rate: 增长率
    out_chans_first_conv: 第一层卷积的卷积核个数
    n_classes: 类别数
'''
class FCDenseNet(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(5, 5, 5, 5, 5),
                 up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48, n_classes=12):
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []


        ## 第一层卷积：3x3 48个卷积核
        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
                  out_channels=out_chans_first_conv, kernel_size=3,
                  stride=1, padding=1, bias=True))
        cur_channels_count = out_chans_first_conv


        ## encoder



        # 加载预训练模型
        
        # dense Block
        self.denseBlocksDown = nn.ModuleList([])
        # Transition Down
        self.transDownBlocks = nn.ModuleList([])
        
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i]))
            cur_channels_count += (growth_rate*down_blocks[i])
            
            # 每一个denseblock后面都有一个skip connection
            skip_connection_channel_counts.insert(0, cur_channels_count)
            
            self.transDownBlocks.append(TransitionDown(cur_channels_count))

        
        ## Bottleneck
       
        self.add_module('bottleneck', Bottleneck(cur_channels_count,
                                     growth_rate, bottleneck_layers))
        
        # transition up 只包含上一个denseblock的输出，不包含上一个denseblock的输入（即x）
        prev_block_channels = growth_rate*bottleneck_layers
        
        cur_channels_count += prev_block_channels

        
        ## decoder
        
        # transition up
        self.transUpBlocks = nn.ModuleList([])
        # denseblock
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks)):
            # transition up
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            # dense block up, 此block的输入为transition up的输出+对应的encoder过程的skip connection 
            self.denseBlocksUp.append(DenseBlock(
                cur_channels_count, growth_rate, up_blocks[i],
                    upsample=True))
            prev_block_channels = growth_rate*up_blocks[i]
            #cur_channels_count += prev_block_channels

        cur_channels_count = growth_rate*up_blocks[-1]

        ## Softmax
        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
               out_channels=n_classes, kernel_size=1, stride=1,
                   padding=0, bias=True)
        self.softmax = nn.LogSoftmax(dim=1)

    # 前向传播
    def forward(self, x):
        out = self.firstconv(x)

        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)
        out = self.softmax(out)
        return out


# FCDenseNet56
def FCDenseNet56(n_classes):
    return FCDenseNet(
        in_channels=3, down_blocks=(4, 4, 4, 4, 4),
        up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=4,
        growth_rate=12, out_chans_first_conv=48, n_classes=n_classes)


# FCDenseNet67
def FCDenseNet67(n_classes, growth_rate=16):
    return FCDenseNet(
        in_channels=3, down_blocks=(5, 5, 5, 5, 5),
        up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
        growth_rate=growth_rate, out_chans_first_conv=48, n_classes=n_classes)


# FCDenseNet103
def FCDenseNet103(n_classes, growth_rate):
    return FCDenseNet(
        in_channels=3, down_blocks=(4, 5, 7, 10, 12),
        up_blocks=(12, 10, 7, 5, 4), bottleneck_layers=15,
        growth_rate=growth_rate, out_chans_first_conv=48, n_classes=n_classes)
