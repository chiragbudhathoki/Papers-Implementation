# MNIST GAN

[![Project image](https://i.pinimg.com/1200x/e5/a7/7d/e5a77dc036cf527723dda4ce28a8752b.jpg)](https://pin.it/2Ig6VcYos)

A PyTorch implementation of a fully connected Generative Adversarial Network (GAN) that learns to generate handwritten digit images from the MNIST dataset.

## How it works

- The **generator** maps 128-dimensional random noise to a flattened 28 × 28 grayscale image.
- The **discriminator** predicts whether an image is real (from MNIST) or generated.
- Both networks are trained adversarially using binary cross-entropy loss and Adam optimizers.
- Training uses a real-image target of `0.9` (one-sided label smoothing) and a fake-image target of `0`.
- The script uses CUDA when available and otherwise runs on the CPU.

## Requirements

- Python 3.9 or later
- PyTorch
- Torchvision
- TensorBoard

Install the dependencies with:

```bash
pip install torch torchvision tensorboard
```

## Run

From the directory containing `GAN_code.py`, run:

```bash
python GAN_code.py
```

The script downloads MNIST automatically into `dataset/` the first time it runs. Training runs for 100 epochs with a batch size of 256. It prints the generator and discriminator losses at the start of each epoch.

## View training progress

Training logs are written beside the script under `runs/GAN_MNIST/`. Start TensorBoard from the project directory:

```bash
tensorboard --logdir runs/GAN_MNIST
```

Open the URL printed by TensorBoard (usually `http://localhost:6006`) to view the generated and real MNIST image grids and the loss scalars.

## Configuration

The main settings are defined near the top of `GAN_code.py`:

| Setting | Default |
| --- | ---: |
| Latent noise dimension | 128 |
| Image size | 28 × 28 grayscale |
| Batch size | 256 |
| Epochs | 100 |
| Discriminator learning rate | 0.0001 |
| Generator learning rate | 0.0003 |

Edit these values in the script to change the training configuration.
