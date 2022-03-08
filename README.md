# BatchFormer: Learning to Explore Sample Relationships for Robust Representation Learning

All Codes in the paper will be coming soon. 

Here is the Pytorch Code of BatchFormer

    def BatchFormer(x, y, encoder, is_training):
      # x: input features with the shape [N, C]
      # encoder: TransformerEncoderLayer(C,4,C,0.5)
      if not is_training:
          return x, y
      pre_x = x
      x = encoder(x.unsqueeze(1)).squeeze(1)
      x = torch.cat([pre_x, x], dim=0)
      y = torch.cat([y, y], dim=0)
      return x, y


If you find this repository helpful, please consider cite:

    @inproceedings{hou2022batch,
    title={BatchFormer: Learning to Explore Sample Relationships for Robust Representation Learning},
    author={Hou, Zhi and Yu, Baosheng and Tao, Dacheng},
    booktitle={CVPR},
    year={2022}
    }

Feel free to contact "zhou9878 at uni dot sydney dot edu dot au" if you have any questions. If you are in ahurry for the code, you can also directly contact him.