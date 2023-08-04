from torch import nn

class Distribution_Model(nn.Module):
    """
    Default LinearNet which has 3 fc layers
    """
    def __init__(self, args, input_size, output_size):
        # input_size : ~1024 * 24, output_size : ~2048
        super().__init__()
        num_layers = args.num_layers
        self.first_layer = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout()
        )
        self.last_layer = nn.Sequential(
            nn.Linear(2048, output_size)
        )
        self.hidden_layers = nn.ModuleList()
        for i in range(1, num_layers-1):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout()
            ))

    def forward(self, x):
        x1 = self.first_layer(x)
        for layer in self.hidden_layers :
            x1 = layer(x1)
        out = self.last_layer(x1)
        
        return out