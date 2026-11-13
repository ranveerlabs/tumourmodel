import torch
import torch.nn as nn

class ConvBlock3D(nn.Module):
    """
    3D Convolutional block with Instance Normalization & LeakyReLU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class TumorSpatialEncoder(nn.Module):
    """
    3D Spatial Encoder:
    - Input: 2 channels [MRI scan, Current tumor mask]
    - Output: Bottleneck 3D spatial feature map
    """
    def __init__(self, in_channels=2, base_filters=16):
        super().__init__()
        self.enc1 = ConvBlock3D(in_channels, base_filters)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc2 = ConvBlock3D(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc3 = ConvBlock3D(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.bottleneck = ConvBlock3D(base_filters * 4, base_filters * 8)

    def forward(self, mri, mask):
        x = torch.cat([mri, mask], dim=1)
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        return self.bottleneck(p3)


if __name__ == "__main__":
    print("Testing 3D Tumor Spatial Encoder...")
    model = TumorSpatialEncoder(in_channels=2, base_filters=16)
    dummy_mri = torch.randn(1, 1, 64, 64, 64)
    dummy_mask = torch.randn(1, 1, 64, 64, 64)
    features = model(dummy_mri, dummy_mask)
    print(f"Success! Output 3D feature map shape: {features.shape}")
