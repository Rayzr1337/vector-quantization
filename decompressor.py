import numpy as np
from PIL import Image
import math
import struct


def decompress(cb_path, stream_path, output_path):
    # Read the codebook
    with open(cb_path, "rb") as f:
        bh, bw, k = struct.unpack(">HHI", f.read(8))
        codebook = np.frombuffer(f.read(), dtype=np.uint8).reshape(k, bh * bw)

    # Read image information and compressed data
    with open(stream_path, "rb") as f:
        H, W, bh, bw, k = struct.unpack(">IIHHI", f.read(16))
        packed = np.frombuffer(f.read(), dtype=np.uint8)

    # Calculate the number of blocks and bits per index
    nbits = max(1, math.ceil(math.log2(k)))
    Hp = math.ceil(H / bh) * bh
    Wp = math.ceil(W / bw) * bw
    num_blocks = (Hp // bh) * (Wp // bw)

    # Decode the block indices
    bits = np.unpackbits(packed)[:num_blocks * nbits]
    bits = bits.reshape(num_blocks, nbits)
    powers = 1 << np.arange(nbits - 1, -1, -1)
    indices = (bits * powers).sum(axis=1)

    # Rebuild the image and remove padding
    blocks = codebook[indices].reshape(num_blocks, bh, bw)
    image = (
        blocks.reshape(Hp // bh, Wp // bw, bh, bw)
        .swapaxes(1, 2)
        .reshape(Hp, Wp)
    )
    image = image[:H, :W].astype(np.uint8)

    Image.fromarray(image, mode="L").save(output_path)
    return output_path