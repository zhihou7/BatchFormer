# BatchFormer: Learning to Explore Sample Relationships for Robust Representation Learning

All Codes in the paper will be coming soon.

Here is the Pytorch Code of BatchFormer

    def BatchFormer(x, y, encoder, is_training):
      # x: input features with the shape [N, C]
      # encoder: TransformerEncoderLayer(C,4,C,0.5) 
      if not is_training:
          return x, y 
      pre x = x
      x = encoder(x.unsqueeze(1)).squeeze(1) 
      x = torch.cat([pre x, x], dim=0)
      y = torch.cat([y, y], dim=0)
      return x, y
