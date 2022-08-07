from jittor import nn, models
import jittor as jt


def start_grad(model):
    for param in model.parameters():
        if 'running_mean' in param.name() or 'running_var' in param.name(): continue
        param.start_grad()

def stop_grad(model):
    for param in model.parameters():
        param.stop_grad()

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        jt.init.gauss_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        jt.init.gauss_(m.weight, 1.0, 0.02)
        jt.init.constant_(m.bias, 0.0)


class ResidualBlock(nn.Module):
    ''' Implements a residual block '''

    def __init__(self, channels: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels, affine=False),

            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels, affine=False),
        )

    def execute(self, x):
        return x + self.layers(x)


class GlobalGenerator(nn.Module):
    ''' Implements the global subgenerator (G1) for transferring styles at lower resolutions '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        fb_blocks: int = 4,  # 3  4
        res_blocks: int = 12, # 9  12
    ):
        super().__init__()

        # Initial convolutional layer
        g1 = [
            nn.ReflectionPad2d(3),
            nn.Conv(in_channels, base_channels, kernel_size=7, padding=0),
            nn.InstanceNorm2d(base_channels, affine=False),
            nn.ReLU(),
        ]

        channels = base_channels
        # Frontend blocks
        for _ in range(fb_blocks):
            g1 += [
                nn.Conv(channels, 2 * channels, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(2 * channels, affine=False),
                nn.ReLU(),
            ]
            channels *= 2

        # Residual blocks
        for _ in range(res_blocks):
            g1 += [ResidualBlock(channels)]

        # Backend blocks
        for _ in range(fb_blocks):
            g1 += [
                nn.ConvTranspose(channels, channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(channels // 2, affine=False),
                nn.ReLU(),
            ]
            channels //= 2

        # Output convolutional layer as its own nn.Sequential since it will be omitted in second training phase
        self.out_layers = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv(base_channels, out_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        )

        self.g1 = nn.Sequential(*g1)

    def execute(self, x):
        x = self.g1(x)
        x = self.out_layers(x)
        return x

# print(GlobalGenerator(3, 3))


class Discriminator(nn.Module):

    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride=2, normalization=True):
            'Returns downsampling layers of each discriminator block'
            layers = [nn.Conv(in_filters, out_filters, 4, stride=stride, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters, eps=1e-05, momentum=0.1, affine=True))
            layers.append(nn.LeakyReLU(scale=0.2))
            return layers

        self.model = nn.Sequential(*discriminator_block((in_channels * 2), 64, normalization=False),
                                   *discriminator_block(64, 128), *discriminator_block(128, 256),
                                   *discriminator_block(256, 512, stride=1), nn.Conv(512, 1, 4, padding=1, bias=False))

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, input):
        return self.model(input)


class VGG19(nn.Module):
    ''' Wrapper for pretrained torchvision.models.vgg19 to output intermediate feature maps '''

    def __init__(self):
        super().__init__()

        vgg_features = models.vgg19(pretrained=True).features

        self.f1 = nn.Sequential(*[vgg_features[x] for x in range(2)])
        self.f2 = nn.Sequential(*[vgg_features[x] for x in range(2, 7)])
        self.f3 = nn.Sequential(*[vgg_features[x] for x in range(7, 12)])
        self.f4 = nn.Sequential(*[vgg_features[x] for x in range(12, 21)])
        self.f5 = nn.Sequential(*[vgg_features[x] for x in range(21, 30)])

        for param in self.parameters():
            param.requires_grad = False

    def execute(self, x):
        h1 = self.f1(x)
        h2 = self.f2(h1)
        h3 = self.f3(h2)
        h4 = self.f4(h3)
        h5 = self.f5(h4)
        return [h1, h2, h3, h4, h5]