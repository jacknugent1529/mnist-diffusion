# Diffusion Model for MNIST
Basic diffusion model for MNIST for the purposes of exploring diffusion models and guidance in particular. 

## Usage
The training can be run with `python train.py [options]`. Use `python train.py --help` to see the options.

Inference can be done in the following manner with the following snippet:
```python
ddpm = DDPM.load_from_checkpoint(<path>, map_location='cpu')

ddpm.eval()

im = ddpm.sample(n_images)
im = (im + 1) / 2 # normalize from -1 to 1 -> 0 to 1 range
```

Training the classifier for classifier-guided diffusion can be done similarly with `mnist_classifier.py`.

## Notes
- unconditional diffusion
    - worked well after training for 25 epochs (~15 minutes)
- classifier guided diffusion
    - classifier trained for 9 epochs (~4 minutes)
- classifier-free diffusion
    - trained for 40 epochs
    - results still less reliable than classifier-guided diffusion
