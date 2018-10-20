## Goal

For this project, I wanted to train a DRAW model using a new dataset of album
art covers:

![Album art covers](examples/test_xs.png)

[Source](https://archive.org/details/audio-covers)

To do this, I used a GCP VM with the following specs:
- 4 vCPUs
- 16 GB memory
- 1 NVIDIA Tesla K80

I ran the `[train_albums.py](train_albums.py)` for 500 epochs with a batch size
of 64, which took ~12 hours and cost ~21US$.

## Training

### Performance Graphs

![Loss graph](examples/loss.png)
![Latent vector histogram](examples/recon_histogram.png)

### Reconstructions

Final:
![Final reconstruction](examples/montage.png)

Epoch 0:
![Epoch 0](examples/manifold_00000000.gif)

Epoch 10000:
![Epoch 10000](examples/manifold_00010000.gif)

Epoch 50928:
![Epoch 10000](examples/manifold_00050928.gif)

Epoch 116832:
![Final reconstruction, animated](examples/manifold_00116832.gif)

## Usage
