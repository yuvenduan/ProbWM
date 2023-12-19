import torch
import torch.nn as nn
import torchvision.models as tvmodels
import torchvision.transforms.functional as F
import displays
from sklearn.decomposition import PCA
import numpy as np

def imagenet_transform(x):
    """
    x: a batch of images, values should be in [0, 1]
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    size = 224
    # See 
    x = F.resize(x, size, antialias=None)
    if not isinstance(x, torch.Tensor):
        x = F.to_tensor(x)
    x = F.normalize(x, mean=mean, std=std)
    return x

def taskonomy_transform(x):
    size = 256
    x = F.resize(x, size, antialias=None)
    x = x * 2 - 1
    return x

def get_cnn(cnn_archi, cnn_pret, cnn_layer='last', skip_last=True, pca_dim=16, exp_names=None, **unused):
    """
    get the cnn model
    Args:
        cnn_archi: architecture of cnn
        cnn_pret: pretraining method of cnn
        skip_last: output of CNN is the last layer if True, 
            else output the second-to-last-layer
    """
    preprocess = None

    if cnn_archi == 'Identity':
        # When embedding is precomputed as input, use identity as 'encoder'
        cnn = nn.Identity()

        if cnn_pret != 'none':
            raise NotImplementedError('Identity can not be pretrained')
    
    elif cnn_archi in ['ResNet-18', 'ResNet-50']:
        if cnn_pret in ['Classification_ImageNet', 'none']:
            if cnn_archi == 'ResNet-18':
                model_handle = tvmodels.resnet18
                model_weights = tvmodels.ResNet18_Weights.IMAGENET1K_V1
            elif cnn_archi == 'ResNet-50':
                model_handle = tvmodels.resnet50
                model_weights = tvmodels.ResNet50_Weights.IMAGENET1K_V2
            else:
                raise NotImplementedError(f'{cnn_archi} not implemented')

            cnn = model_handle(weights=None)
            if cnn_pret != 'none':
                assert cnn_pret == 'Classification_ImageNet'
                cnn = model_handle(weights=model_weights)
                preprocess = imagenet_transform

            if skip_last:
                cnn.fc = nn.Identity()
        else:
            raise NotImplementedError('No pretrained model')

    elif cnn_archi == 'AlexNet':
        cnn = tvmodels.alexnet(weights=None)
        if cnn_pret != 'none':
            assert cnn_pret == 'Classification_ImageNet'
            cnn = tvmodels.alexnet(weights=tvmodels.AlexNet_Weights.IMAGENET1K_V1)
            preprocess = imagenet_transform

        if skip_last:
            cnn.classifier[-1] = nn.Identity()

    elif cnn_archi == 'ViT-B':
        if cnn_pret == 'none':
            cnn = tvmodels.vit_b_32(weights=None)
        elif cnn_pret == 'Classification_ImageNet':
            cnn = tvmodels.vit_b_32(weights=tvmodels.ViT_B_32_Weights.IMAGENET1K_V1)
            preprocess = imagenet_transform
        else:
            raise NotImplementedError('No pretrained model')

        if skip_last and cnn_pret in ['none', 'Classification_ImageNet']:
            cnn.heads[-1] = nn.Identity()
        
    else:
        raise NotImplementedError('CNN not implemented')
    
    class Wrapper(nn.Module):
        def __init__(self, cnn):
            super().__init__()
            self.cnn: nn.Module = cnn
            self.pca: PCA = None
            self.embedding = None

            def hook(module, input, output: torch.Tensor):
                self.embedding = output.reshape(output.shape[0], -1).detach().cpu().numpy()
                if self.pca is not None:
                    self.embedding = self.pca.transform(self.embedding)

            if cnn_layer == 'last':
                self.cnn.register_forward_hook(hook)
            elif cnn_layer == 'layer1':
                self.cnn.layer1.register_forward_hook(hook)
            elif cnn_layer == 'layer2':
                self.cnn.layer2.register_forward_hook(hook)
            elif cnn_layer == 'layer3':
                self.cnn.layer3.register_forward_hook(hook)
            elif cnn_layer == 'layer4':
                self.cnn.layer4.register_forward_hook(hook)
            else:
                raise NotImplementedError('Layer not supported')

        def forward(self, x):
            if preprocess is not None:
                x = preprocess(x)
            self.cnn(x).reshape(x.shape[0], -1)
            return self.embedding
        
    cnn = Wrapper(cnn)
    cnn.requires_grad_(False)
    cnn.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn.to(device)

    if pca_dim is not None:
        pca = PCA(n_components=pca_dim)
        full_embeddings = []

        if exp_names is None:
            exp_names = ['RedBlue', 'BlackWhite', 'ColoredSquares']
        for exp_name in exp_names:
            exp = displays.get_experiment(exp_name)

            imgs = []
            for i in range(len(exp)):
                trial = exp.get_trial(i)
                display = trial['displays'][1]
                imgs.append(display)
            
            imgs = torch.stack(imgs).to(device)
            embeddings = cnn(imgs)
            full_embeddings.append(embeddings)

        full_embeddings = np.concatenate(full_embeddings, axis=0)
        pca.fit(full_embeddings)
        cnn.pca = pca

    return cnn