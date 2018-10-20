## Goal

For this project, I wanted to train a DRAW model using a new dataset of album
art covers:

![Album art covers](examples/test_xs.png)

[Source](https://archive.org/details/audio-covers)

To do this, I used a GCP VM with the following specs:
- 4 vCPUs
- 16 GB memory
- 1 NVIDIA Tesla K80

I ran the [train_albums.py](train_albums.py) for 500 epochs with a batch size
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

New albums to reconstruct:

![Albums](examples/experiment_montage.png)

Reconstruction:

![Animated reconstruction](examples/experiment_recon.gif)

See [DRAW Experiments](DRAW\ Experiments.ipynb) for the experiment notebook.

### Observations

The "brush stroke" paints in a somewhat boring top-left to bottom-right
direction at a 45deg angle, possibly as a result of the number of timesteps
being too small for the canvas size and Gaussians in the filterbank, forcing
the network to be overly parsimonious in its use of the brush during the
limited time it has to reconstruct the image.
