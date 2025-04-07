"""
Memory-efficient U-Net model for manuscript inpainting with proper channel handling.
This version carefully manages channel dimensions throughout the network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Double convolution block with residual connection."""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Only use residual connection if input and output channels match
        self.use_residual = (in_channels == out_channels)
        
    def forward(self, x):
        conv_out = self.conv(x)
        if self.use_residual:
            return conv_out + x
        return conv_out

class UNetInpaint(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        """
        Memory-efficient U-Net for manuscript inpainting with proper channel handling.
        
        Args:
            in_channels (int): Number of input channels (RGB + Mask = 4)
            out_channels (int): Number of output channels (RGB = 3)
        """
        super(UNetInpaint, self).__init__()
        
        # Define feature dimensions for each level
        # Using smaller feature sizes to save memory
        self.features = [32, 64, 128, 256]
        
        # Initial convolution
        self.inc = DoubleConv(in_channels, self.features[0])
        
        # Encoder (downsampling) path
        self.down_convs = nn.ModuleList()
        for i in range(len(self.features) - 1):
            self.down_convs.append(nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(self.features[i], self.features[i + 1])
            ))
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(self.features[-1], self.features[-1]),
            nn.ConvTranspose2d(self.features[-1], self.features[-1], kernel_size=2, stride=2)
        )
        
        # Decoder (upsampling) path
        # Note: input channels are doubled due to skip connections
        self.up_convs = nn.ModuleList()
        self.up_trans = nn.ModuleList()
        
        for i in range(len(self.features) - 1, 0, -1):
            # Upsampling convolution
            self.up_trans.append(
                nn.ConvTranspose2d(self.features[i], self.features[i-1], kernel_size=2, stride=2)
            )
            # Convolution after concatenation
            # Input channels = current level features + same level features from encoder
            self.up_convs.append(
                DoubleConv(self.features[i-1] * 2, self.features[i-1])
            )
        
        # Final convolution
        self.outc = nn.Sequential(
            nn.Conv2d(self.features[0], out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Initial convolution
        x1 = self.inc(x)
        
        # Encoder path with skip connections
        skip_connections = [x1]
        x = x1
        
        for down_conv in self.down_convs:
            x = down_conv(x)
            skip_connections.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        skip_connections = skip_connections[:-1]  # Remove last skip connection (same as bottleneck output)
        skip_connections = skip_connections[::-1]  # Reverse for decoder path
        
        for i in range(len(self.up_convs)):
            # Upscale current features
            x = self.up_trans[i](x)
            
            skip = skip_connections[i]
            
            # Handle potential size mismatch
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
            
            # Concatenate skip connection
            x = torch.cat([skip, x], dim=1)
            
            # Apply convolutions
            x = self.up_convs[i](x)
        
        # Final convolution
        return self.outc(x)

def test_model():
    """Test the model with dummy data and print intermediate shapes."""
    model = UNetInpaint()
    x = torch.randn((1, 4, 1000, 1000))
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    out = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Output value range: [{out.min().item():.3f}, {out.max().item():.3f}]")
    
    return out.shape == (1, 3, 1000, 1000)

if __name__ == "__main__":
    test_model()