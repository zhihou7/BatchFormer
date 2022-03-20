import torch
import torch.optim as optim

from models.image_extractor import get_image_extractor
from models.visual_product import VisualProductNN
from models.manifold_methods import RedWine, LabelEmbedPlus, AttributeOperator
from models.modular_methods import GatedGeneralNN
from models.graph_method import GraphFull
from models.symnet import Symnet
from models.compcos import CompCos

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TransformerDecorator(torch.nn.Module):
    def __init__(self, model, model_name, add_bt=0, eval_bt=0):
        super(TransformerDecorator, self).__init__()
        self.model = model
        self.model_name = model_name
        self.add_bt = add_bt
        self.eval_bt = eval_bt

        if self.add_bt:
            self.encoder_layers = torch.nn.TransformerEncoderLayer(512, 4, 512, 0.5)

    @property
    def is_open(self):
        return self.model.is_open

    @is_open.setter
    def is_open(self, value):
        self.model.is_open = value

    def update_feasibility(self, value):
        self.model.update_feasibility(value)

    def forward_feats(self, feature):
        feature[0] = feature[0].view(feature[0].size(0), -1)

        old_feat = feature[0]
        if self.model.training and self.add_bt or self.eval_bt:
            feature[0] = feature[0].unsqueeze(1)
            feature[0] = self.encoder_layers(feature[0])
            feature[0] = feature[0].squeeze(1)
            if self.add_bt > 1:
                feature[0] = torch.cat([old_feat, feature[0]], dim=0)
        return feature

    def forward(self, feature):
        feature = self.forward_feats(feature)
        feature[0] = self.model(feature)

        return feature[0]


def configure_model(args, dataset):
    image_extractor = None
    is_open = False

    if args.model == 'visprodNN':
        model = VisualProductNN(dataset, args)
    elif args.model == 'redwine':
        model = RedWine(dataset, args)
    elif args.model == 'labelembed+':
        model = LabelEmbedPlus(dataset, args)
    elif args.model == 'attributeop':
        model = AttributeOperator(dataset, args)
    elif args.model == 'tmn':
        model = GatedGeneralNN(dataset, args, num_layers=args.nlayers, num_modules_per_layer=args.nmods)
    elif args.model == 'symnet':
        model = Symnet(dataset, args)
    elif args.model == 'graphfull':
        model = GraphFull(dataset, args)
    elif args.model == 'compcos':
        model = CompCos(dataset, args)
        if dataset.open_world and not args.train_only:
            is_open = True
    else:
        raise NotImplementedError

    # add Transformer
    model = TransformerDecorator(model, args.model, args.add_bt)
    model = model.to(device)

    if args.update_features:
        print('Learnable image_embeddings')
        image_extractor = get_image_extractor(arch = args.image_extractor, pretrained = True)
        image_extractor = image_extractor.to(device)

    # configuring optimizer
    if args.model=='redwine':
        optim_params = filter(lambda p: p.requires_grad, model.parameters())
    elif args.model=='attributeop':
        attr_params = [param for name, param in model.named_parameters() if 'attr_op' in name and param.requires_grad]
        other_params = [param for name, param in model.named_parameters() if 'attr_op' not in name and param.requires_grad]
        optim_params = [{'params':attr_params, 'lr':0.1*args.lr}, {'params':other_params}]
    elif args.model=='tmn':
        gating_params = [
            param for name, param in model.named_parameters()
            if 'gating_network' in name and param.requires_grad
        ]
        network_params = [
            param for name, param in model.named_parameters()
            if 'gating_network' not in name and param.requires_grad
        ]
        optim_params = [
            {
                'params': network_params,
            },
            {
                'params': gating_params,
                'lr': args.lrg
            },
        ]
    else:
        model_params = [param for name, param in model.named_parameters() if param.requires_grad and not name.__contains__('encoder_layers')]
        optim_params = [{'params':model_params}]

        if args.add_bt:
            model_params = [param for name, param in model.named_parameters() if param.requires_grad and name.__contains__('encoder_layers')]
            optim_params.append({'params':model_params, 'lr': args.lrb})

    if args.update_features:
        ie_parameters = [param for name, param in image_extractor.named_parameters()]
        optim_params.append({'params': ie_parameters,
                            'lr': args.lrg})
    optimizer = optim.Adam(optim_params, lr=args.lr, weight_decay=args.wd)

    model.is_open = is_open

    return image_extractor, model, optimizer