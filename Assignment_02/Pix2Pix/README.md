# Pix2Pix

This folder contains a compact Pix2Pix implementation for the facades dataset.

## Data

The dataset should use the common paired-image facades format, where each image is split into left and right halves:

```text
datasets/facades/
  train/*.jpg
  test/*.jpg
```

The facades dataset was downloaded locally to:

```text
../../facades/
```

It contains 400 training images, 100 validation images and 106 test images.

## Training

```bash
python train.py --data_root ../../facades --output_dir runs/facades_full --epochs 20 --batch_size 4 --image_size 256 --preview_every 5 --save_every 10 --device cuda
```

Outputs:

```text
runs/facades_full/
  previews/
  checkpoints/
```

The epoch 20 preview was copied to:

```text
../pics/pix2pix_facades_epoch_0020.png
```

## Implementation

- `FCN_network.py`: U-Net style fully convolutional generator and PatchGAN discriminator.
- `facades_dataset.py`: paired facades image loader.
- `train.py`: adversarial loss, L1 loss, preview saving and checkpoint saving.
