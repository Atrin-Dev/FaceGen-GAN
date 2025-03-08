class critic(nn.Module):
    def __init__(self, d_dim=16):
        super(Critic, self).__init__()

        self.crit = nn.Sequential(
            # Conv2d: in_channels, out_channels, kernel_size, stride=1, padding=0
            ## New width and height: # (n+2*pad-ks)//stride +1
            nn.Conv2d(3, d_dim, 4, 2, 1),  # (n+2*pad-ks)//stride +1 = (128+2*1-4)//2+1=64x64 (ch: 3,16)
            nn.InstanceNorm2d(d_dim),
            nn.LeakyReLU(0.2),

            nn.Conv2d(d_dim, d_dim * 2, 4, 2, 1),  ## 32x32 (ch: 16, 32)
            nn.InstanceNorm2d(d_dim * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(d_dim * 2, d_dim * 4, 4, 2, 1),  ## 16x16 (ch: 32, 64)
            nn.InstanceNorm2d(d_dim * 4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(d_dim * 4, d_dim * 8, 4, 2, 1),  ## 8x8 (ch: 64, 128)
            nn.InstanceNorm2d(d_dim * 8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(d_dim * 8, d_dim * 16, 4, 2, 1),  ## 4x4 (ch: 128, 256)
            nn.InstanceNorm2d(d_dim * 16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(d_dim * 16, 1, 4, 1, 0),  # (n+2*pad-ks)//stride +1=(4+2*0-4)//1+1= 1X1 (ch: 256,1)

        )

    def forward(self, image):
        # image: 128 x 3 x 128 x 128
        crit_pred = self.crit(image)  # 128 x 1 x 1 x 1
        return crit_pred.view(len(crit_pred), -1)  ## 128 x 1