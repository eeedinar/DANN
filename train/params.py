from models import models

# utility params
fig_mode = None
embed_plot_epoch=10

# model params
use_gpu = True
dataset_mean = (0.5, 0.5, 0.5)
dataset_std = (0.5, 0.5, 0.5)

batch_size = 1024
epochs = 5
gamma = 10
theta = 1

# path params
data_root = './data'

mnist_path = data_root + '/MNIST'
mnistm_path = data_root + '/MNIST_M'
svhn_path = data_root + '/SVHN'
syndig_path = data_root + '/SynthDigits'
usps_path = data_root + '/USPS'

save_dir = './experiment'


# specific dataset params
extractor_dict = {'MNIST_MNIST_M': models.Extractor(),
                  'MNIST_SVHN'   : models.Extractor(),
                  'USPS_MNIST'   : models.Extractor(),
                  'SVHN_MNIST'   : models.Extractor(),
                  'SVHN_USPS'    : models.Extractor(),
                  }

class_dict = {'MNIST_MNIST_M': models.Class_classifier(),
              'MNIST_SVHN'   : models.Class_classifier(),
              'USPS_MNIST'   : models.Class_classifier(),
              'SVHN_MNIST'   : models.Class_classifier(),
              'SVHN_USPS'    : models.Class_classifier(),
              }

domain_dict = {'MNIST_MNIST_M': models.Domain_classifier(),
               'MNIST_SVHN'   : models.Domain_classifier(),
               'USPS_MNIST'   : models.Domain_classifier(),
               'SVHN_MNIST'   : models.Domain_classifier(),
               'SVHN_USPS'    : models.Domain_classifier(),
              }
