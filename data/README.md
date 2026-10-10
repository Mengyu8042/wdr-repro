# Data dictionary

This file describes the datasets, their roles in the experiments, and the
pretrained model checkpoints used in the image experiments. 

## Simulated data

The simulation studies in Figures 1--5 use no external data. Every dataset is 
generated inside the seeded drivers in `code/simulation/` (linear Gaussian and 
spatial trajectories, observations, and the Kalman oracle references).

## Image datasets

[MNIST](https://keras.io/api/datasets/mnist/) contains handwritten digits 
from 0 to 9. [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) contains
photographs in ten categories: airplane, automobile, bird, cat, deer, dog,
frog, horse, ship, and truck. Each observation is an image paired with its
class label.

| Dataset | Image format | Classes | Training images | Test images | Total images |
|---|---|---:|---:|---:|---:|
| MNIST | 28 x 28 pixels, grayscale | 10 | 60,000 | 10,000 | 70,000 |
| CIFAR-10 | 32 x 32 pixels, RGB | 10 | 50,000 | 10,000 | 60,000 |

### Use in the experiments

The image experiments in Figures 6--7 concern class-conditional image generation: 
the class label specifies the images to generate. For Frechet Inception Distance 
(FID) evaluation, `code/image_exp/run_image.py` loads the training split of each
dataset through `torchvision` and selects the first 2,000 images of
each class in dataset order. This gives 20,000 real reference images for an
experiment across all ten classes. The datasets are downloaded automatically
into `code/image_exp/data/` when FID evaluation is enabled.

### Fields and preprocessing

| Field | MNIST | CIFAR-10 |
|---|---|---|
| Image | 28 x 28 grayscale pixels | 32 x 32 pixels with RGB channels |
| Pixel value | Unsigned 8-bit intensity, 0--255 | Unsigned 8-bit intensity per channel, 0--255 |
| Label | Integer 0--9 denoting the digit | Integer 0--9 denoting, in order, airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck |

`torchvision` handles the dataset downloads and archive formats. In 
`build_real_pool_mnist` and `build_real_pool_cifar10`, the `ToTensor` transform
converts the image pixels to floating-point tensors with values in [0, 1].

## Pretrained model files

`code/image_exp/download_ckpts.py` downloads the following files from the Hugging 
Face repository `Mengyu8042/wdr-checkpoints` to `code/image_exp/models/`:

- `uncond_model_mnist.pt` and `uncond_model_cifar10.pt`: diffusion-model
  parameters used to generate images.
- `classifier_mnist.pth.tar` and `classifier_cifar10.pth.tar`: classifier
  parameters used for class conditioning and accuracy evaluation.

The package also includes `vae_mnist_z32.pt` and `vae_cifar10_z32.pt` in
`code/image_exp/models/checkpoints/`. These provide 32-dimensional image
representations for the WDR and T-WDR resampling steps.
