from src.config import Config
from src.utils import create_dir
import numpy as np

# do your thing with the hyper-parameters
from src.edge_connect import EdgeConnect


def randomTune(config):
    # 'LR': 0.0001,                   # learning rate
    # 'D2G_LR': 0.1,                  # discriminator/generator learning rate ratio
    # 'BETA1': 0.0,                   # adam optimizer beta1
    # 'BETA2': 0.9,                   # adam optimizer beta2
    # 'L1_LOSS_WEIGHT': 1,            # l1 loss weight
    # 'FM_LOSS_WEIGHT': 10,           # feature-matching loss weight
    config.MAX_EPOCHES = 1
    # config.MAX_STEPS = 3
    experiments = 50
    for i in range(experiments):
        # sample from a Uniform distribution on a log-scale
        config.LR = 10 ** np.random.uniform(-2, -5)  # Sample learning rate candidates in the range (0.01 to 0.00001)
        config.D2G_LR = 10 ** np.random.uniform(-1, -3)  # Sample regularization candidates in the range (0.001 to 0.1)
        config.PATH = './checkpoints/places2_tune_%d_%f%f_' % (i, config.LR, config.D2G_LR)
        # logdir= config.PATH+('/log_%s_%s' % (config.LR , config.D2G_LR))
        create_dir(config.PATH)

        model = EdgeConnect(config)
        # config.print()
        print('\nEx ;%d learning_rate:%f  D_Learning_rate: %f:\n' % (i, config.LR, config.D2G_LR))

        model.train()
