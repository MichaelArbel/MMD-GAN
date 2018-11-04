__all__= ['model', 'wgan_gp', 'cramer', 'ops', 'mmd', 'resnet', 'architecture']


def model_class(name):
    name = name.lower()
    if name == 'mmd':
        from .model import MMD_GAN as Model
    elif name == 'gan':
        from .gan import GAN as Model
    elif name == 'wgan_gp':
        from .wgan_gp import WGAN_GP as Model
    elif name == 'cramer':
        from .cramer import Cramer_GAN as Model
    elif name == 'smmd':
        from .smmd import SMMD as Model
    elif name == 'swgan':
        from .smmd import SWGAN as Model
    elif name == 'sobolevgan_gp':
        from .sobolevgan_gp import SobolevGAN as Model
    elif name in {'sobolevmmd', 'gcmmd'}:
        from .sobolevmmd import KernelSobolevMMD_GAN as Model
    else:
        raise ValueError("unknown model {}".format(name))
    return Model
