from torch import nn

class AutoEncoderConv1d(nn.Module):
    def __init__(self, num_channel):
        # one parameter: number of channels
        # this project needs three channels
        super(AutoEncoderConv1d, self).__init__()
        self.num_channel = num_channel
        # 128 96 256
        num1 = 6
        num2 = 12
        num3 = 24
        self.encoder = nn.Sequential(
            nn.Conv1d(self.num_channel,
                      num1,
                      3,
                      stride=1 #,padding=(1, 1)
                      ),
            nn.BatchNorm1d(num1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(num1,
                      num2, 3,
                      stride=1#, padding=(1, 1)
                      ),
            nn.BatchNorm1d(num2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(feature_num, feature_num_x2,
            #           2, stride=1, padding=(1, 1)),
            # nn.BatchNorm2d(feature_num_x2),
            # nn.LeakyReLU(0.2, inplace=True),
            #
            # nn.Conv2d(feature_num_x2, feature_num_x2,
            #           2, stride=1, padding=(1, 1)),
            # nn.BatchNorm2d(feature_num_x2),
            # nn.LeakyReLU(0.2, inplace=True)
        )

        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(feature_num_x2, feature_num_x2,
            #                    2, stride=1
            #                    #, padding=(1, 1),
            #                    #output_padding=(1, 1)
            #                    ),
            # nn.BatchNorm2d(feature_num_x2),
            # nn.LeakyReLU(0.2, inplace=True),
            #
            # nn.ConvTranspose2d(feature_num_x2, feature_num,
            #                    2, stride=1
            #                    #, padding=(1, 1),
            #                    #output_padding=(1, 1)
            #                    ),
            # nn.BatchNorm2d(feature_num),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(num2, num1,
                               3, stride=1
                               # , padding=(1, 1),
                               # output_padding=(1, 1)
                               ),
            nn.BatchNorm1d(num1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(num1, self.num_channel,
                               3, stride=1
                               # , padding=(1, 1),
                               # output_padding=(0, 1)
                               )
        )

    def forward(self, x):
        f = self.encoder(x)
        out = self.decoder(f)
        return out