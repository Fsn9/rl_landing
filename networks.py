from torch import nn

class OneLayerMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.in_dim = in_dim

        self.out_dim = out_dim

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

class TwoLayerMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.in_dim = in_dim

        self.out_dim = out_dim

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

class neural_network(nn.Module):
    '''
    Feedforward neural network with variable number
    of hidden layers and ReLU nonlinearites
    '''

    def __init__(self,
                layers,
                # layers[0] = input layer
                # layers[i] = # of neurons at i-th layer
                # layers[-1] = output layer
                dropout=False,
                p_dropout=0.2,
                ):
        super(neural_network,self).__init__()

        self.network_layers = []
        n_layers = len(layers)
        for i,neurons_in_current_layer in enumerate(layers[:-1]):
            
            self.network_layers.append(nn.Linear(neurons_in_current_layer,
                                                layers[i+1]) )
            
            if dropout:
                self.network_layers.append( nn.Dropout(p=p_dropout) )

            if i < n_layers - 2:
                self.network_layers.append( nn.ReLU() )
        
        self.network_layers = nn.Sequential(*self.network_layers)

    def forward(self,x):
        for layer in self.network_layers:
            x = layer(x)
        return x
