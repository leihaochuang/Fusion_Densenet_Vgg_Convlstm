from torch import nn
from layers import DenseBlock, TransitionDown, Bottleneck, TransitionUp
import torchvision.models
from fusion_config import args_setting
import torch

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
    def __init__(self, in_channels=3, down_blocks=(6, 12, 24, 16),
                 up_blocks=(32, 32, 12, 6), bottleneck_layers=4,
                 growth_rate=16, out_chans_first_conv=64, n_classes=12):
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks

        cur_channels_count = 0
        skip_connection_channel_counts = []
        args = args_setting()

        ## 第一层卷积：3x3 48个卷积核
        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
                  out_channels=out_chans_first_conv, kernel_size=3,
                  stride=1, padding=1, bias=True))
        cur_channels_count = out_chans_first_conv


        ## encoder


        assert (
            args.layers in [121, 169, 161, 201]
        ), 'Only layers 121, 169, 161 and 201are defined, but got {}'.format(
            args.layers)
        # 加载预训练模型
        pretrained_model = torchvision.models.densenet121(pretrained=False)
        pthfile = './pretrained/densenet121-a639ec97.pth'
        pretrained_dict = torch.load(pthfile)
        model_dict = pretrained_model.state_dict()
        pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
        model_dict.update(pretrained_dict_1)
        pretrained_model.load_state_dict(model_dict)
        #初始化denseblock和transition
        self.denseBlocksDown = nn.ModuleList([])
        # Transition Down
        self.transDownBlocks = nn.ModuleList([])
        # densenet预训练模型的第一个块
        self.denseBlocksDown.append(pretrained_model._modules['features'][4])
        cur_channels_count += (growth_rate * down_blocks[0])
        skip_connection_channel_counts.insert(0, cur_channels_count)
        self.transDownBlocks.append(pretrained_model._modules['features'][5])
        cur_channels_count //= 2
        # densenet预训练模型的第二个块
        self.denseBlocksDown.append(pretrained_model._modules['features'][6])
        cur_channels_count += (growth_rate * down_blocks[1])
        skip_connection_channel_counts.insert(0, cur_channels_count)
        self.transDownBlocks.append(pretrained_model._modules['features'][7])
        cur_channels_count //= 2
        # densenet预训练模型的第三个块
        self.denseBlocksDown.append(pretrained_model._modules['features'][8])
        cur_channels_count += (growth_rate * down_blocks[2])
        skip_connection_channel_counts.insert(0, cur_channels_count)
        self.transDownBlocks.append(pretrained_model._modules['features'][9])
        cur_channels_count //= 2
        # rdensenet预训练模型的第四个块
        self.denseBlocksDown.append(pretrained_model._modules['features'][10])
        cur_channels_count += (growth_rate * down_blocks[3])
        skip_connection_channel_counts.insert(0, cur_channels_count)
        self.transDownBlocks.append(pretrained_model._modules['features'][9])
        cur_channels_count //= 2
        del pretrained_model  # clear memory

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


def FCDenseNet121(n_classes, growth_rate):
    return FCDenseNet(
        in_channels=3, down_blocks=(6, 12, 24, 16),
        up_blocks=(16, 24, 12, 6), bottleneck_layers=32,
        growth_rate=growth_rate, out_chans_first_conv=64, n_classes=n_classes)

