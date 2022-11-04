import torch.nn


class Conv2d_AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(

            torch.nn.Linear(11520, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 9)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(9, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 11520),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = torch.flatten(x)
        print(x.shape)
        encoder = self.encoder(x)
        print(encoder.shape)
        decoder = self.decoder(encoder)
        return torch.reshape(decoder, (10, 3, 128, 3))
