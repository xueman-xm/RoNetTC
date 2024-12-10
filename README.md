# Reliable Open-Set Network Traffic Classification

## Requirment
- Pytorch 1.3.0
- Python 3
- sklearn
- numpy
- scipy


## Python train.py

## Input Data Format
The model inputs are: parallel inputs for three views. For example, in one view, `X` represents the raw bytes of multiple packets in a flow.

Here, `H`, `W`, and `C` refer to the height, width, and channels of `X`, respectively. These are formed by embedding the original one-dimensional bytes and stacking them into a channel in a cross-shaped, non-collapsed manner with a preset configuration of four packets.
- **Dimensions**: `(H, W, C)`
  - `H`: Height
  - `W`: Width
  - `C`: Channels, 

