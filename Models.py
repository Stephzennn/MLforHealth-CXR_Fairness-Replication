import torch
import torch.nn as nn

class ERM(nn.Module):
    def __init__(self, num_classes, hparams):
        super().__init__()
        self.network = nn.Sequential(
            models.get_featurizer(hparams),
            models.get_clf_head(hparams, models.get_featurizer(hparams).n_outputs, num_classes)
        )
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["lr"],
            weight_decay=hparams['weight_decay']
        )

    def update(self, minibatches, device):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = nn.CrossEntropyLoss()(self.predict(all_x), all_y.squeeze().long())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)
    

# This is a streamlined version of the class from the original paper, offering a more efficient execution.
class CBR_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, max_pool=True):
        super(CBR_Block, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        if max_pool:
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)
    

# This class closely resembles the one in the paper. 
# The specific composition of the model is hard-coded, and its origin is unclear. 
# Therefore, it's best to retain the current structure.

# It uses the CBR_Block class as a building block to create a hierarchical structure with an output dimension of 512.


class CBR(nn.Module):
    def __init__(self):
        super(CBR, self).__init__()
        self.model = nn.Sequential(
            CBR_Block(3, 32, 7),
            CBR_Block(32, 64, 7),
            CBR_Block(64, 128, 7),
            CBR_Block(128, 256, 7),
            CBR_Block(256, 512, 7, max_pool=False)
        )
        self.emb_dim = 512
        
    def forward(self, x):
        return self.model(x)
