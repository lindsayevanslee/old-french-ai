import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Applies two consecutive convolutional layers with Batch Normalization and ReLU activation."""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)

class UNetInpaint(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, features=[64, 128, 256, 512]):
        """
        Initializes the U-Net model.
        
        Args:
            in_channels (int): Number of input channels (e.g., RGB + Mask = 4).
            out_channels (int): Number of output channels (e.g., RGB = 3).
            features (list): List of feature sizes for the encoder.
        """
        super(UNetInpaint, self).__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder: Downsampling path
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        # Decoder: Upsampling path
        # Use a copy of the features list to avoid modifying the original list
        reversed_features = features[::-1]
        
        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for feature in reversed_features:
            self.upconvs.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(feature*2, feature))
        
        # Final Convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        skip_connections = skip_connections[::-1]
        for idx in range(len(self.upconvs)):
            x = self.upconvs[idx](x)
            skip_connection = skip_connections[idx]
            
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])
            
            # Concatenate along the channel dimension
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx](concat_skip)
        
        return self.final_conv(x)

if __name__ == "__main__":
    # Test the model with a dummy input of size 1000x1000
    model = UNetInpaint()
    x = torch.randn((1, 4, 1000, 1000))  # Batch size 1, 4 channels (RGB + Mask), 1000x1000
    out = model(x)
    print(out.shape)  # Expected Output: torch.Size([1, 3, 1000, 1000])
