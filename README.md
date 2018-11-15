# Model v1

TBD, legacy...

# Model v2

## Data

Lines are encoded to 14-dim vector.

Sample images encodings have 17-dim vectors for each pixel:
- [1]: indicates with 1 whether there's a line on that pixel
- [2-3]: indicates line center distance from current pixel, from -1:+1
- [4-17]: 14-dim centered-line embedding, values -1:+1, tanh activation used

## Files

- Generated lines, all centered, for autoencoder training:
    - input & output: line_originals_v2_392x28x28.npy
- Generated lines with shifted variants, for encoder training:
    - input: line_samples_v2_7234x28x28x1.npy
    - output: samples: line_encodings_v2_7234x28x28x16.npy

## Weights

- Autoencoder weights, 400 epochs, 60k params:
    - lines_autoencoder_v2-385-0.0047.hdf5
- Encoder weights
    - lines_encoded_v2-020-0.000044.hdf5
