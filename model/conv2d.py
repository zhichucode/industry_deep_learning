import torch
from torch import nn

class AutoEncoderConv2d(nn.Module):

    def __init__(self, chnum_in):
        # one parameter: number of channels
        # this project needs three channels
        super(AutoEncoderConv2d, self).__init__()
        self.chnum_in = chnum_in
        feature_num = 128
        feature_num_2 = 96
        feature_num_x2 = 256
        self.encoder = nn.Sequential(
            nn.Conv2d(self.chnum_in,
                      feature_num_2,
                      (3, 3),
                      stride=(1, 2),
                      padding=(1, 1)),
            nn.BatchNorm2d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_num_2,
                      feature_num, (3, 3),
                      stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_num, feature_num_x2,
                      (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_num_x2, feature_num_x2,
                      (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_num_x2, feature_num_x2,
                               (3, 3), stride=(2, 2), padding=(1, 1),
                               output_padding=(1, 1)),
            nn.BatchNorm2d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(feature_num_x2, feature_num,
                               (3, 3), stride=(2, 2), padding=(1, 1),
                               output_padding=(1, 1)),
            nn.BatchNorm2d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(feature_num, feature_num_2,
                               (3, 3), stride=(2, 2), padding=(1, 1),
                               output_padding=(1, 1)),
            nn.BatchNorm2d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(feature_num_2, self.chnum_in,
                               (3, 3), stride=(1, 2), padding=(1, 1),
                               output_padding=(0, 1))
        )

    def forward(self, x):
        f = self.encoder(x)
        out = self.decoder(f)
        return out