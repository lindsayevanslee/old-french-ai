"""
Enhanced U-Net model for manuscript inpainting with attention mechanisms.
The model takes advantage of pre-computed masks and uses attention to better
handle the specific challenges of medieval manuscript inpainting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    """Attention mechanism for focusing on relevant features during inpainting."""
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, 1)
        self.key = nn.Conv2d(in_channels, in_channels//8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Create query, key, value projections
        proj_query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, height * width)
        proj_value = self.value(x).view(batch_size, -1, height * width)
        
        # Calculate attention
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        return self.gamma * out + x

class DoubleConv(nn.Module):
    """Double convolution block with residual connection."""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Add residual connection if input and output channels match
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        return self.conv(x) + self.residual(x)

class UNetInpaint(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, features=[64, 128, 256, 512]):
        """
        Enhanced U-Net for manuscript inpainting.
        
        Args:
            in_channels (int): Number of input channels (RGB + Mask = 4)
            out_channels (int): Number of output channels (RGB = 3)
            features (list): Feature dimensions for each level
        """
        super(UNetInpaint, self).__init__()
        
        # Encoder path
        self.encoder = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        
        # Initial convolution to process the input
        self.initial_conv = DoubleConv(in_channels, features[0])
        
        # Encoder blocks with attention
        in_channels = features[0]
        for feature in features[1:]:
            self.encoder.append(nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, feature)
            ))
            self.attention_blocks.append(AttentionBlock(feature))
            in_channels = feature
            
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(features[-1], features[-1] * 2),
            AttentionBlock(features[-1] * 2),
            nn.ConvTranspose2d(features[-1] * 2, features[-1], kernel_size=2, stride=2)
        )
        
        # Decoder path
        self.decoder = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        
        for feature in reversed(features[1:]):
            self.upconvs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(feature * 2, feature))
            
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], features[0] // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0] // 2, out_channels, kernel_size=1),
            nn.Sigmoid()  # Ensure output is in [0,1] range
        )
        
    def forward(self, x):
        # Store skip connections
        skip_connections = []
        
        # Initial convolution
        x = self.initial_conv(x)
        skip_connections.append(x)
        
        # Encoder path
        for i, (enc, attn) in enumerate(zip(self.encoder, self.attention_blocks)):
            x = enc(x)
            x = attn(x)
            skip_connections.append(x)
            
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with skip connections
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        
        for i, (upconv, dec) in enumerate(zip(self.upconvs, self.decoder)):
            x = upconv(x)
            
            # Handle potential size mismatch
            if x.shape != skip_connections[i].shape:
                x = F.interpolate(x, size=skip_connections[i].shape[2:])
                
            # Concatenate skip connection
            concat_skip = torch.cat((skip_connections[i], x), dim=1)
            x = dec(concat_skip)
            
        # Final convolution
        return self.final_conv(x)

def test_model():
    """Test the model with dummy data."""
    model = UNetInpaint()
    x = torch.randn((1, 4, 1000, 1000))  # Batch size 1, 4 channels (RGB + Mask), 1000x1000
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output value range: [{out.min().item():.3f}, {out.max().item():.3f}]")
    
if __name__ == "__main__":
    test_model()