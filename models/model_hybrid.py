import torch
import torch.nn as nn
from models.unet import UNet
from models.pvt2 import pvt_v2_b2, pvt_v2_b2_flow
from models.unet_dec import UNetDec


class AnomalyDetection(nn.Module):
    def __init__(self, config, logger, is_trained=True):
        super().__init__()
        self.number_of_layers = config.MODEL.CONCATENATED_LAYERS
        self.reconstruction = UNet(config=config)

        self.encoder_of = pvt_v2_b2_flow()
        self.encoder_fr = pvt_v2_b2()
        self.decoder = UNetDec(number_frames=[5, 5, 5, 5, 5, 5])

        # Initialize
        if is_trained:
            self.encoder_of.init_weights(logger, config.MODEL.PRETRAINED)
            self.encoder_fr.init_weights(logger, config.MODEL.PRETRAINED)

    def forward(self, fr, of):
        # reconstruction
        out_of = self.reconstruction(of)

        # Prediction
        condition = self.encoder_of(out_of)
        features = []
        for f in fr:
            features.append(self.encoder_fr(f))

        features.append(condition)

        # Transpose list of lists => https://stackoverflow.com/questions/6473679/transpose-list-of-lists
        features = list(map(list, zip(*features)))

        for i in range(len(features)):
            feature_i = features[i]
            if i not in self.number_of_layers:
                feature_i = feature_i[:-1]

            features[i] = torch.cat([fea for fea in feature_i], dim=1)

        out_fr = self.decoder(features)

        return out_fr, out_of


