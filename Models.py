import torch
import torch.nn as nn

import torch.nn.functional as F
from torchvision import datasets, models, transforms


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

# This class is a modification of the class of the same name in the paper. It takes the embedding type, a bool of either pretrained or not and concat features 
# and returns based on the embedtype 
class EmbModel(nn.Module):
    def __init__(self, emb_type, pretrain, concat_features=0):
        super(EmbModel, self).__init__()
        self.emb_type = emb_type
        self.concat_features = concat_features

        if emb_type == 'densenet':
            model = models.densenet121(pretrained=pretrain)
            self.encoder = nn.Sequential(*list(model.features)[:-1])
            self.emb_dim = model.classifier.in_features
            '''
            Pretrained=True: Downloads and loads a DenseNet121 model with weights pretrained on the ImageNet dataset.
            Pretrained=False: Initializes a DenseNet121 model with random weights (untrained).
            '''
        elif emb_type == 'resnet':
            model = models.resnet50(pretrained=pretrain)
            self.encoder = nn.Sequential(*list(model.children())[:-2])
            self.emb_dim = model.fc.in_features

            '''
            Pretrained=True: Downloads and loads a ResNet50 model with weights pretrained on the ImageNet dataset.
            Pretrained=False: Initializes a ResNet50 model with random weights (untrained).
            '''
        elif emb_type == 'vision_transformer':
            self.encoder = timm.create_model('vit_deit_small_patch16_224', pretrained=pretrain, num_classes=0)
            self.emb_dim = self.encoder.num_features

            '''
            Pretrained=True: Downloads and loads a Vision Transformer model (ViT) with weights pretrained on the ImageNet dataset.
            Pretrained=False: Initializes a Vision Transformer model with random weights (untrained).
            '''

        elif emb_type == 'CBR':
            self.encoder = CBR()
            self.emb_dim = self.encoder.emb_dim

            '''
            Pretrained=True/False: The custom CBR implementation does not inherently support pretrained weights, so this flag would not affect its initialization unless we modify the CBR class to support pretrained weights.
            '''
        self.n_outputs = self.emb_dim + concat_features

    def forward(self, inp):
        x = inp['img'] if isinstance(inp, dict) else inp
        x = self.encoder(x).squeeze()

        if self.emb_type == 'densenet':
            x = F.relu(x)
            x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)

        if isinstance(inp, dict):
            x = torch.cat([x, inp['concat']], dim=-1)

        return x

#Here is a simplified version of the get_featurizer function. We might not use it since we will likely stick to one set of hyperparameters.

def get_featurizer(hparams):
    n_concat_features = 1 if hparams['concat_group'] and hparams['protected_attr'] == 'sex' else len(Constants.group_vals[hparams['protected_attr']]) if hparams['concat_group'] else 0
    return EmbModel(hparams['model'], pretrain=True, concat_features=n_concat_features)

#This is a simplified version of the get_clf_head function, which creates a classification head based on the input parameters.


def get_clf_head(hparams, featurizer_n_outputs, n_classes):
    n_hidden = featurizer_n_outputs // hparams['clf_head_ratio']
    return nn.Sequential(
        nn.Linear(featurizer_n_outputs, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_classes)
    )
