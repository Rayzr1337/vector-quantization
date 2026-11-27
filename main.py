import os
import struct

from compressor import compress
from decompressor import decompress

def main():
    while True:
        print("\n=== Image Compressor ===")
        print("1. Compress an image")
        print("2. Decompress an image")
        print("3. Quit")
        choice = input("Choose an option: ").strip()

        if choice == "3":
            print("Goodbye.")
            break

        if choice == "1":
            img_path = input("Image path: ").strip()
            if not os.path.isfile(img_path):
                print("Error: file not found.")
                continue

            try:
                bh = int(input("Block height: "))
                bw = int(input("Block width: "))
                k = int(input("Codebook size k: "))
            except ValueError:
                print("Error: block height, block width and k must be whole numbers.")
                continue

            if bh < 1 or bw < 1 or k < 1:
                print("Error: all values must be at least 1.")
                continue

            try:
                cb_path, stream_path, ratio, ratio_headers = compress(img_path, bh, bw, k)
            except (OSError, ValueError) as e:
                print(f"Compression failed: {e}")
                continue

            original_bytes = os.path.getsize(img_path)
            compressed_bytes = os.path.getsize(cb_path) + os.path.getsize(stream_path)

            print("\n--- Compression complete ---")
            print("Codebook file:", cb_path)
            print("Stream file:  ", stream_path)
            print(f"Original size:   {original_bytes:,} bytes")
            print(f"Compressed size: {compressed_bytes:,} bytes (.cb + .bin)")
            print(f"Compression ratio (raw 8-bit pixels): {ratio:.2f}")
            print(f"Compression ratio (with headers):     {ratio_headers:.2f}")
            print(f"Compression ratio (file sizes):       {original_bytes / compressed_bytes:.2f}")
            
        elif choice == "2":
            cb_path = input("Codebook file (.cb): ").strip()
            if not os.path.isfile(cb_path):
                print("Error: codebook file not found.")
                continue

            stream_path = input("Stream file (.bin): ").strip()
            if not os.path.isfile(stream_path):
                print("Error: stream file not found.")
                continue

            output_path = input("Output image name (with extension, e.g. out.png): ").strip()
            if not os.path.splitext(output_path)[1]:
                print("Error: output name needs an extension like .png or .jpg.")
                continue

            try:
                out = decompress(cb_path, stream_path, output_path)
            except (OSError, ValueError, IndexError, struct.error) as e:
                print(f"Decompression failed: {e}")
                continue

            compressed_bytes = os.path.getsize(cb_path) + os.path.getsize(stream_path)
            decompressed_bytes = os.path.getsize(out)

            print("\n--- Decompression complete ---")
            print("Image saved to:", out)
            print(f"Compressed size:   {compressed_bytes:,} bytes (.cb + .bin)")
            print(f"Decompressed size: {decompressed_bytes:,} bytes")
            print(f"Expansion ratio:   {decompressed_bytes / compressed_bytes:.2f}")
            print("Note: lossy algorithm, so this will not match the original image.")

        else:
            print("Error: please enter 1, 2 or 3.")

if __name__ == "__main__":
    main()