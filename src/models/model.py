# thanks to: https://github.com/lucidrains/byol-pytorch/blob/master/byol_pytorch/byol_pytorch.py
import torch
import torchvision.models
from torch import nn
import copy
import collections
import src.optimizers.loss as losses


def load_model(config, model_name, checkpoint_path=None):
    model = None

    if model_name == 'BYOL':
        model = BYOL(config)
    elif model_name == 'Downstream':
        model = DownstreamNetwork(config)

    if checkpoint_path is not None:
        print('>> load checkppoints ...')
        device = torch.device('cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    return model


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


class MLPNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)


class Encoder(nn.Module):
    def __init__(self, vision_model_name='resnet50', pre_trained=True):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential()
        if vision_model_name == 'resnet50':
            self.encoder.add_module(
                "encoder_layer",
                nn.Sequential(
                    torchvision.models.resnet50(pretrained=pre_trained).conv1,
                    torchvision.models.resnet50(pretrained=pre_trained).bn1,
                    torchvision.models.resnet50(pretrained=pre_trained).relu,
                    torchvision.models.resnet50(pretrained=pre_trained).maxpool,
                    torchvision.models.resnet50(pretrained=pre_trained).layer1,
                    torchvision.models.resnet50(pretrained=pre_trained).layer2,
                    torchvision.models.resnet50(pretrained=pre_trained).layer3,
                    torchvision.models.resnet50(pretrained=pre_trained).layer4,
                    torchvision.models.resnet50(pretrained=pre_trained).avgpool,
                )
            )
        elif vision_model_name == 'resnet34':
            self.encoder.add_module(
                "encoder_layer",
                nn.Sequential(
                    torchvision.models.resnet34(pretrained=pre_trained).conv1,
                    torchvision.models.resnet34(pretrained=pre_trained).bn1,
                    torchvision.models.resnet34(pretrained=pre_trained).relu,
                    torchvision.models.resnet34(pretrained=pre_trained).maxpool,
                    torchvision.models.resnet34(pretrained=pre_trained).layer1,
                    torchvision.models.resnet34(pretrained=pre_trained).layer2,
                    torchvision.models.resnet34(pretrained=pre_trained).layer3,
                    torchvision.models.resnet34(pretrained=pre_trained).layer4,
                    torchvision.models.resnet34(pretrained=pre_trained).avgpool,
                )
            )
        elif vision_model_name == 'resnet18':
            self.encoder.add_module(
                "encoder_layer",
                nn.Sequential(
                    torchvision.models.resnet18(pretrained=pre_trained).conv1,
                    torchvision.models.resnet18(pretrained=pre_trained).bn1,
                    torchvision.models.resnet18(pretrained=pre_trained).relu,
                    torchvision.models.resnet18(pretrained=pre_trained).maxpool,
                    torchvision.models.resnet18(pretrained=pre_trained).layer1,
                    torchvision.models.resnet18(pretrained=pre_trained).layer2,
                    torchvision.models.resnet18(pretrained=pre_trained).layer3,
                    torchvision.models.resnet18(pretrained=pre_trained).layer4,
                    torchvision.models.resnet18(pretrained=pre_trained).avgpool,
                )
            )
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = self.encoder(x)
        out = self.flatten(out)
        return out


class BYOL(nn.Module):
    def __init__(self, config):
        super(BYOL, self).__init__()
        self.config = config
        self. target_ema_updater = EMA(beta=config['ema_decay'])
        self.online_encoder = Encoder(vision_model_name=config['vision_model_name'], pre_trained=config['pre_trained'])
        self.online_projector = MLPNetwork(input_dim=config['input_dim'],
                                           output_dim=config['output_dim'], hidden_dim=config['hidden_dim'])
        self.online_predictor = MLPNetwork(input_dim=config['hidden_dim'],
                                           output_dim=config['output_dim'], hidden_dim=config['hidden_dim'])

        self.target_encoder = None
        self.target_projector = None

    def get_target_network(self):
        self.target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(self.target_encoder, False)
        self.target_projector = copy.deepcopy(self.online_projector)
        set_requires_grad(self.target_projector, False)

    def update_target_network(self):
        if self.target_encoder is not None and self.target_projector is not None:
            update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)
            update_moving_average(self.target_ema_updater, self.target_projector, self.online_projector)
        else:
            self.get_target_network()

    def forward(self, x1, x2):
        out_encoder_x1 = self.online_encoder(x1)
        out_encoder_x2 = self.online_encoder(x2)

        out_projector_x1 = self.online_projector(out_encoder_x1)
        out_projector_x2 = self.online_projector(out_encoder_x2)

        out_predictor_x1 = self.online_predictor(out_projector_x1)
        out_predictor_x2 = self.online_predictor(out_projector_x2)

        with torch.no_grad():
            if self.target_encoder is None or self.target_projector is None:
                self.get_target_network()

            out_target_encoder_x1 = self.target_encoder(x1)
            out_target_encoder_x2 = self.target_encoder(x2)
            out_target_projector_x1 = self.target_projector(out_target_encoder_x1)
            out_target_projector_x2 = self.target_projector(out_target_encoder_x2)

        loss01 = losses.loss_fn(out_predictor_x1, out_target_projector_x2.detach())
        loss02 = losses.loss_fn(out_predictor_x2, out_target_projector_x1.detach())

        loss = loss01 + loss02

        return loss.mean()


class DownstreamNetwork(nn.Module):
    def __init__(self, config):
        super(DownstreamNetwork, self).__init__()
        self.encoder = None
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.class_num = config['class_num']
        self.classifier = nn.Sequential(
            collections.OrderedDict(
                [
                    ('linear01-1', nn.Linear(self.input_dim, self.hidden_dim)),
                    ('bn01', nn.BatchNorm1d(self.hidden_dim)),
                    ('act01', nn.ReLU()),
                    ('linear01-2', nn.Linear(self.hidden_dim, self.hidden_dim)),

                    ('linear02-1', nn.Linear(self.hidden_dim, self.hidden_dim)),
                    ('bn02', nn.BatchNorm1d(self.hidden_dim)),
                    ('act02', nn.ReLU()),
                    ('linear02-2', nn.Linear(self.hidden_dim, self.hidden_dim)),

                    ('linear03-1', nn.Linear(self.hidden_dim, self.hidden_dim)),
                    ('bn03', nn.BatchNorm1d(self.hidden_dim)),
                    ('act03', nn.ReLU()),
                    ('linear03-2', nn.Linear(self.hidden_dim, self.class_num)),
                ]
            )
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.classifier(out)
        return out


if __name__ == '__main__':
    test_config = {
        "vision_model_name": "resnet34",
        "pre_trained": True,
        "ema_decay":0.99,
        "input_dim": 512,
        "hidden_dim": 4096,
        "output_dim": 4096,
        "class_num": 10
    }
    test_encoder = BYOL(config=test_config)

    input_data = torch.rand(2, 3, 512, 512)
    output_data = test_encoder.online_encoder(input_data)

    test_downstream = DownstreamNetwork(config=test_config, encoder = test_encoder.online_encoder)
    output_data = test_downstream(input_data)

    print(output_data.size())

