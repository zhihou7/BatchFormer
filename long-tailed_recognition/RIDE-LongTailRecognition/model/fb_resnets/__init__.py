import torch

def obtain_global_models(add_bt, output_dim):
    heads = 4 if output_dim > 128 else 1
    if add_bt == 12:
        encoder_layers = torch.nn.TransformerEncoderLayer(output_dim, heads, output_dim, 0.5)
        encoder_layers = torch.nn.TransformerEncoder(encoder_layers, 8)
    elif add_bt == 13:
        encoder_layers = torch.nn.TransformerEncoderLayer(output_dim, heads, output_dim, 0.5)
        encoder_layers = torch.nn.TransformerEncoder(encoder_layers, 6)
    else:
        encoder_layers = torch.nn.TransformerEncoderLayer(output_dim, heads, output_dim, 0.5)
    return encoder_layers
