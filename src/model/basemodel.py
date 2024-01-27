from torch import nn, Tensor


class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def get_number_parameters(self) -> int:
        """Return the number of parameters of the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_only_learned_parameters(self) -> dict[str, Tensor]:
        """ save only the learned parameters """
        state_dict: dict[str, Tensor] = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                state_dict[name] = param

        return state_dict


class BaseModel(Model):
    def __init__(self,
                 hidden_size: int,
                 last_layer: bool=True,
                 num_classes: int=2
                 ) -> None:
        super().__init__()
        self.last_layer = last_layer
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        self.last_linear = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.relu = nn.ReLU()
    
    def put_last_layer(self, last_layer: bool) -> None:
        self.last_layer = last_layer
    
    def forward_last_layer(self, x: Tensor) -> Tensor:
        """ forward data in relu and last fc """
        x = self.relu(x)
        x = self.last_linear(x)
        return x
    
    def get_hidden_size(self) -> int:
        """ get the hidden_size (before last fc layer) """
        return self.hidden_size
    