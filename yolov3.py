import torch
import torch.nn as nn
from const import device, ANCHORS, s, class_labels

# Modify CNNBlock to include dropout
class CNNBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, use_batch_norm=True, dropout_rate=0.1, **kwargs): 
        super().__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not use_batch_norm, **kwargs) 
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.01)  # Reduced momentum for better generalization
        self.activation = nn.LeakyReLU(0.1) 
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.use_batch_norm = use_batch_norm 

    def forward(self, x): 
        x = self.conv(x) 
        if self.use_batch_norm: 
            x = self.bn(x) 
            x = self.activation(x)
            x = self.dropout(x)
        return x

# Defining residual block
class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        
        # Defining all the layers in a list and adding them based on number of
        # repeats mentioned in the design
        res_layers = []
        for _ in range(num_repeats):
            res_layers += [
                nn.Sequential(
                    nn.Conv2d(channels, channels // 2, kernel_size=1),
                    nn.BatchNorm2d(channels // 2),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.LeakyReLU(0.1)
                )
            ]
        self.layers = nn.ModuleList(res_layers)
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            residual = x
            x = layer(x)
            if self.use_residual:
                x = x + residual
        return x


# Modify ScalePrediction to include L2 regularization
class ScalePrediction(nn.Module): 
    def __init__(self, in_channels, num_classes, dropout_rate=0.1): 
        super().__init__() 
        self.pred = nn.Sequential( 
            nn.Conv2d(in_channels, 2*in_channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(2*in_channels, momentum=0.01),
            nn.Dropout2d(p=dropout_rate),
            nn.LeakyReLU(0.1), 
            nn.Conv2d(2*in_channels, (num_classes + 5) * 3, kernel_size=1),
        ) 
        self.num_classes = num_classes 
    
    # Defining the forward pass and reshaping the output to the desired output 
    # format: (batch_size, 3, grid_size, grid_size, num_classes + 5) 
    def forward(self, x): 
        output = self.pred(x) 
        output = output.view(x.size(0), 3, self.num_classes + 5, x.size(2), x.size(3)) 
        output = output.permute(0, 1, 3, 4, 2) 
        return output


# Modify YOLOv3 initialization
class YOLOv3(nn.Module): 
    def __init__(self, in_channels=3, num_classes=20, dropout_rate=0.1): 
        super().__init__() 
        self.num_classes = num_classes 
        self.in_channels = in_channels 

        # Apply weight initialization
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

        # Layers list for YOLOv3 with dropout
        self.layers = nn.ModuleList([ 
            CNNBlock(in_channels, 32, kernel_size=3, stride=1, padding=1, dropout_rate=dropout_rate), 
            CNNBlock(32, 64, kernel_size=3, stride=2, padding=1, dropout_rate=dropout_rate), 
            ResidualBlock(64, num_repeats=1), 
            CNNBlock(64, 128, kernel_size=3, stride=2, padding=1, dropout_rate=dropout_rate), 
            ResidualBlock(128, num_repeats=2), 
            CNNBlock(128, 256, kernel_size=3, stride=2, padding=1, dropout_rate=dropout_rate), 
            ResidualBlock(256, num_repeats=8), 
            CNNBlock(256, 512, kernel_size=3, stride=2, padding=1, dropout_rate=dropout_rate), 
            ResidualBlock(512, num_repeats=8), 
            CNNBlock(512, 1024, kernel_size=3, stride=2, padding=1, dropout_rate=dropout_rate), 
            ResidualBlock(1024, num_repeats=4), 
            CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0, dropout_rate=dropout_rate), 
            CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1, dropout_rate=dropout_rate), 
            ResidualBlock(1024, use_residual=False, num_repeats=1), 
            CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0, dropout_rate=dropout_rate), 
            ScalePrediction(512, num_classes=num_classes, dropout_rate=dropout_rate), 
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0, dropout_rate=dropout_rate), 
            nn.Upsample(scale_factor=2), 
            CNNBlock(768, 256, kernel_size=1, stride=1, padding=0, dropout_rate=dropout_rate), 
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1, dropout_rate=dropout_rate), 
            ResidualBlock(512, use_residual=False, num_repeats=1), 
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0, dropout_rate=dropout_rate), 
            ScalePrediction(256, num_classes=num_classes, dropout_rate=dropout_rate), 
            CNNBlock(256, 128, kernel_size=1, stride=1, padding=0, dropout_rate=dropout_rate), 
            nn.Upsample(scale_factor=2), 
            CNNBlock(384, 128, kernel_size=1, stride=1, padding=0, dropout_rate=dropout_rate), 
            CNNBlock(128, 256, kernel_size=3, stride=1, padding=1, dropout_rate=dropout_rate), 
            ResidualBlock(256, use_residual=False, num_repeats=1), 
            CNNBlock(256, 128, kernel_size=1, stride=1, padding=0, dropout_rate=dropout_rate), 
            ScalePrediction(128, num_classes=num_classes, dropout_rate=dropout_rate) 
        ]) 

        # Apply weight initialization
        self.apply(init_weights)
    
    # Forward pass for YOLOv3 with route connections and scale predictions
    def forward(self, x):
        outputs = []
        route_connections = []

        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue
            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)
            
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
        return outputs

    def to(self, device_):
        super().to(device_)
        self.device = device_
        return self


# Testing YOLO v3 model 
if __name__ == "__main__": 
    # Setting number of classes and image size 
    num_classes = 20
    IMAGE_SIZE = 416

    # Creating model and testing output shapes 
    model = YOLOv3(num_classes=num_classes) 
    x = torch.randn((1, 3, IMAGE_SIZE, IMAGE_SIZE)) 
    out = model(x) 
    print(out[0].shape) 
    print(out[1].shape) 
    print(out[2].shape) 

    # Asserting output shapes 
    assert model(x)[0].shape == (1, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5) 
    assert model(x)[1].shape == (1, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5) 
    assert model(x)[2].shape == (1, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5) 
    print("Output shapes are correct!")
