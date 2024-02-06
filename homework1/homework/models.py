import torch
import torch.nn.functional as F


class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        """
        Your code here

        Compute mean(-log(softmax(input)_label))

        @input:  torch.Tensor((B,C))
        @target: torch.Tensor((B,), dtype=torch.int64)

        @return:  torch.Tensor((,))

        Hint: Don't be too fancy, this is a one-liner
        """
        
        # Use cross-entropy built in function
        return F.cross_entropy(input, target)


class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Input size
        input_size = 3 * 64 * 64

        # Output features
        output_features = 6
        
        # Create linear layer
        self.linear = torch.nn.Linear(input_size, output_features)

    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """

        # Get batch size
        batch_size = x.size(0)

        # Flatten tensor
        x = x.view(batch_size, -1)

        # Pass through linear layer
        output = self.linear(x)

        return output


class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Input size
        input_size = 3*64*64

        # Output features
        output_features = 6

        # Hidden layer size
        hidden_layer_size = 128

        # Define layers of MLP
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_layer_size), # First linear layer
            torch.nn.ReLU(), # Hidden layer
            torch.nn.Linear(hidden_layer_size, output_features) # Second linear layer
        )

    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """

        # Get batch size
        batch_size = x.size(0)

        # Flatten tensor
        x = x.view(batch_size, -1)

        # Pass through layers
        output = self.layers(x)

        return output


model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
