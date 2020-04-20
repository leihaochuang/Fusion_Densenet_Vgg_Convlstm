import fusion_config
from layers import DenseBlock, TransitionDown, Bottleneck, TransitionUp
from utils import *
from Densenet_pre import FCDenseNet121
from torchvision import models


# VGG+Densenet -ConvLSTM混合网络模型

def generate_model(args):
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    assert args.model in ['Densenet-Vgg', 'Densenet']
    if args.model == 'Densenet-Vgg':
        model = Vgg_Dense_Convlstm_Net67(fusion_config.class_num, fusion_config.growth_rate).to(device)
    else:
        model = FCDenseNet121(fusion_config.class_num, fusion_config.growth_rate).to(device)
    return model
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

# 定义Dense_net网络


class FCDense_Vgg_Net(nn.Module):
    def __init__(self, in_channels=3, cloud_channels=1, down_blocks=(5, 5, 5, 5, 5),
                 up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48, n_classes=2, fusion_channels_count=1392):
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        self.fusion_channels_count = fusion_channels_count
        cur_channels_count = 0
        skip_connection_channel_counts = []

        # 第一层卷积：3x3 48个卷积核
        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
                                               out_channels=out_chans_first_conv, kernel_size=3,
                                               stride=1, padding=1, bias=True))
        cur_channels_count = out_chans_first_conv

        # encoder

        # dense Block
        self.denseBlocksDown = nn.ModuleList([])
        # Transition Down
        self.transDownBlocks = nn.ModuleList([])

        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i]))
            cur_channels_count += (growth_rate * down_blocks[i])

            # 每一个denseblock后面都有一个skip connection
            skip_connection_channel_counts.insert(0, cur_channels_count)

            self.transDownBlocks.append(TransitionDown(cur_channels_count))

        # Bottleneck

        self.add_module('bottleneck', Bottleneck(self.fusion_channels_count,
                                                 growth_rate, bottleneck_layers))

        # transition up 只包含上一个denseblock的输出，不包含上一个denseblock的输入（即x）
        prev_block_channels = growth_rate * bottleneck_layers

        cur_channels_count += prev_block_channels

        # decoder

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
            prev_block_channels = growth_rate * up_blocks[i]
            # cur_channels_count += prev_block_channels

        cur_channels_count = growth_rate * up_blocks[-1]
        self.convlstm = ConvLSTM(input_size=(3, 12),
                                 input_dim=80,
                                 hidden_dim=[80, 80],
                                 kernel_size=(3, 3),
                                 num_layers=2,
                                 batch_first=False,
                                 bias=True,
                                 return_all_layers=False)
        ## Softmax
        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
                                   out_channels=n_classes, kernel_size=1, stride=1,
                                   padding=0, bias=True)
        self.softmax = nn.LogSoftmax(dim=1)
    #  Vgg16 编码器 处理点云数据
        self.inc = inconv(cloud_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)

    # 前向传播
    def forward(self, x):

        x_image = x['image']
        x_cloud = x['point_cloud_image']
        x_image = torch.unbind(x_image, dim=1)
        x_cloud = torch.unbind(x_cloud, dim=1)


        # test 点云融合效果

        for item in range(len(x_image)):
            out_image = self.firstconv(x_image[item])
            skip_connections = []
            # print(x_cloud[item].shape)
            # exit()
            x1 = self.inc(x_cloud[item])
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            out_cloud_fusion = x1, x2, x3, x4, x5
            # for iii in out_cloud_fusion:
            #     print(iii.shape)

            for i in range(len(self.down_blocks)):
                out_image = self.denseBlocksDown[i](out_image)
                # 现在传递给解码器的是图像数据浅层特征 如有需要也可也传点云
                skip_connections.append(out_image)

                # 特征图缩小一半
                out_image = self.transDownBlocks[i](out_image)
                # print(out_image.shape)
                # 融合点云和图像数据通道上
                if i < 4:
                    out = torch.cat([out_image, out_cloud_fusion[i+1]], dim=1)



            out = self.bottleneck(out)

        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)
        out = self.softmax(out)
        return out


# FCDenseNet67

def Vgg_Dense_Convlstm_Net67(n_classes, growth_rate=16):
    return FCDense_Vgg_Net(
        in_channels=3, down_blocks=(5, 5, 5, 5, 5),
        up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
        growth_rate=growth_rate, out_chans_first_conv=48, n_classes=n_classes)
def Vgg_Dense_Convlstm_Net103(n_classes, growth_rate):
    return FCDense_Vgg_Net(
        in_channels=3, down_blocks=(4, 5, 7, 10, 12),
        up_blocks=(12, 10, 7, 5, 4), bottleneck_layers=15,
        growth_rate=growth_rate, out_chans_first_conv=48, n_classes=n_classes)




