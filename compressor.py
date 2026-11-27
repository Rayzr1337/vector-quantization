import numpy as np 
from PIL import Image
import math, os, struct

def compress(img_path, bh, bw, k):
    img = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
    H, W = img.shape

    pad_h = (-H) % bh
    pad_w = (-W) % bw
    padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode="edge")
    Hp, Wp = padded.shape

    blocks = (padded.reshape(Hp // bh, bh, Wp // bw, bw)
              .swapaxes(1, 2).reshape(-1, bh * bw))

    blocks = blocks.astype(np.float64)
    dim = bh * bw
    rng = np.random.default_rng(0)

    codebook = blocks.mean(axis=0, keepdims=True)
    counts = np.array([len(blocks)])

    while len(codebook) < k:
         #split: all codewords, or only the most populated ones if k isn't a power of 2
        n_split = min(len(codebook), k - len(codebook))
        chosen = np.argsort(-counts)[:n_split]
        keep = np.setdiff1d(np.arange(len(codebook)), chosen)

        c = codebook[chosen]
        is_int = c == np.floor(c)
        lo = np.where(is_int, c - 1, np.floor(c))
        hi = np.where(is_int, c + 1, np.ceil(c))
        codebook = np.vstack([codebook[keep], lo, hi])

        # 1 pass per level, until "no change" at the last level
        max_iter = 100 if len(codebook) == k else 1
        prev_idx = None
        for _ in range(max_iter):
            # assign each block to nearest codeword (L1), in chunks to save memory
            idx = np.empty(len(blocks), dtype=np.int64)
            chunk = max(1, int(2e7 // (len(codebook) * dim)))
            for s in range(0, len(blocks), chunk):
                b = blocks[s:s + chunk]
                d = np.abs(b[:, None, :] - codebook[None, :, :]).sum(axis=2)
                idx[s:s + chunk] = d.argmin(axis=1)

            if prev_idx is not None and np.array_equal(idx, prev_idx):
                break                                   # no block changed group -> stop
            prev_idx = idx

            # average: each codeword = mean of its blocks
            counts = np.bincount(idx, minlength=len(codebook))
            nonempty = counts > 0
            for j in range(dim):
                sums = np.bincount(idx, weights=blocks[:, j], minlength=len(codebook))
                codebook[nonempty, j] = sums[nonempty] / counts[nonempty]

            # empty clusters: reseed from a random block of the biggest cluster
            for e in np.where(~nonempty)[0]:
                big = counts.argmax()
                codebook[e] = blocks[rng.choice(np.where(idx == big)[0])]

    codebook_u8 = np.clip(np.rint(codebook), 0, 255).astype(np.uint8)
    cb_f = codebook_u8.astype(np.float64)

    indices = np.empty(len(blocks), dtype=np.int64)
    chunk = max(1, int(2e7 // (k * dim)))
    for s in range(0, len(blocks), chunk):
        b = blocks[s:s + chunk]
        d = np.abs(b[:, None, :] - cb_f[None, :, :]).sum(axis=2)
        indices[s:s + chunk] = d.argmin(axis=1)
   
    nbits = max(1, math.ceil(math.log2(k)))
    as_bytes = indices.astype(">u4").view(np.uint8).reshape(-1, 4)
    bits = np.unpackbits(as_bytes, axis=1)[:, 32 - nbits:]
    packed = np.packbits(bits.reshape(-1)).tobytes()

    prefix = os.path.splitext(img_path)[0]
    cb_path = prefix + ".cb"
    stream_path = prefix + ".bin"

    with open(cb_path, "wb") as f:
        f.write(struct.pack(">HHI", bh, bw, k))
        f.write(codebook_u8.tobytes())

    with open(stream_path, "wb") as f:
        f.write(struct.pack(">IIHHI", H, W, bh, bw, k))
        f.write(packed)

    #final compression ration calculation and return
    original_size_bits = H * W * 8                      # original image, 8 bits per pixel
    codebook_size_bits = k * bh * bw * 8                # k codewords, 8 bits per pixel each
    labels_size_bits = len(blocks) * nbits              # one label per block
    compressed_size_bits = codebook_size_bits + labels_size_bits

    compression_ratio = original_size_bits / compressed_size_bits

    file_size_bits = (os.path.getsize(cb_path) + os.path.getsize(stream_path)) * 8
    compression_ratio_with_headers = original_size_bits / file_size_bits

    return cb_path, stream_path, compression_ratio, compression_ratio_with_headers

