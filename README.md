## Pytoch_DANN
> The original pytorch code is derived from [CuthbertCai][1]. 
> This is an implementation of [Domain-Adversarial Training of Neural Networks][2]  
> with a GRL(Gradient Reveral Layer). 

### requirements
> python 3.6.2  
> `pip install -r requirements.txt`

### Data
> USPS, MNIST, SVHN will be download from the torch dataset. 
> MINIST_M dataset can be downloaded at [Yaroslav Ganin's homepage][3]. 
> Then you can extract the file to your data directory and run the `preprocess.py` 
> to make the directory able to be used with  
> `torchvision.datasets.ImageFolder`:
```
python preprocess.py
```

### Run on PC (CPU only)
> go to train/params.py and change and then run
```
use_gpu=False
python3 main.py --source_domain USPS --target_domain MNIST --save_dir ./USPS_MNIST --max_epoch 100
```

### Run on Google Colab (Copy directory in your Google Drive)
> run `pytorch_colab.ipynb`

### Run on Discovery Cluster (Copy directory in your Google Drive)
> run the followig:
```
sbatch discovery_job_SVHN_MNIST.script
sbatch discovery_job_SVHN_USPS.script
sbatch discovery_job_USPS_MNIST.script
```
> Note: GPU:t4 should be used to be compatible with pytorch and cuda/11.0



[1]:https://arxiv.org/pdf/1505.07818.pdf
[2]:https://github.com/CuthbertCai/pytorch_DANN
[3]:http://yaroslav.ganin.net/
