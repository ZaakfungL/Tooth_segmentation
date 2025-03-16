import segmentation_models_pytorch as smp
from models.unet import UNet

def select_model(model_name, in_channels, classes):
    if model_name == 'unet':
        return UNet(num_classes=classes)
    elif model_name == 'unetplusplus':
        return smp.UnetPlusPlus(encoder_name="resnet34",
                               encoder_weights="imagenet",
                               in_channels=in_channels,
                               classes=classes,
                               activation='sigmoid')
    elif model_name == 'segformer':
        return smp.Segformer(encoder_name="mit_b0",
                            encoder_weights="imagenet",
                            in_channels=in_channels,
                            classes=classes,
                            activation='sigmoid')
    else:
        raise ValueError(f"Unknown model name: {model_name}")