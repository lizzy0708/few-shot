"""
make_gadf_fixed.py — 2026-10-08. Does NOT modify processed/make_gadf.py. New file
only. Duplicate of make_gadf.py with ONLY the uint8 conversion line changed, per
the wrap-around-sensitivity request: the original does `(gaf_img*255).astype(
np.uint8)` with no clip/offset, which wraps negative GAF values (gaf_img in
[-1,1] for method='difference') around to the high end instead of mapping them
toward 0 (confirmed in the 9/28 audit). This version uses the symmetric [-1,1] ->
[0,255] linear remap: round((v+1)/2*255), clipped to [0,255].

Everything else -- normalization (min-max to [-1,1] before GAF), PAA (implicit
inside pyts' GramianAngularField), GAF computation, window range (same boundary
excluding the final partial window: range(0, len(signal)-window_size,
window_size)), stride, filename convention -- is byte-for-byte identical to
make_gadf.py. Default --save_dir changed to "processed_fixed" (new folder, never
overwrites "processed/").

Usage:
  conda run -n torch python processed/make_gadf_fixed.py
"""
import os
import re
import argparse
import numpy as np
import scipy.io as sio
from pyts.image import GramianAngularField
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="HUST bearing dataset")
    parser.add_argument("--save_dir", type=str, default="processed_fixed")
    parser.add_argument("--domain_mode", type=str, default="coarse",
                        choices=["coarse", "fine"])
    parser.add_argument("--window_size", type=int, default=1024)
    parser.add_argument("--image_size", type=int, default=128)
    return parser.parse_args()


def get_domain(file_name, domain_mode):
    number = re.findall(r'\d+', file_name)[0]
    if domain_mode == "coarse":
        return number[:1] + "00"
    if domain_mode == "fine":
        return number
    raise ValueError(f"Unknown domain_mode: {domain_mode}")


def main():
    args = parse_args()

    gaf = GramianAngularField(image_size=args.image_size, method='difference')
    os.makedirs(args.save_dir, exist_ok=True)

    print("===================================")
    print("GADF preprocessing (wrap-around FIXED)")
    print("===================================")
    print("Data dir   :", args.data_dir)
    print("Save dir   :", args.save_dir)
    print("Domain mode:", args.domain_mode)
    print("Window size:", args.window_size)
    print("Image size :", args.image_size)
    print("===================================")

    for file in os.listdir(args.data_dir):
        if not file.endswith(".mat"):
            continue

        path = os.path.join(args.data_dir, file)
        data = sio.loadmat(path)

        signal = None
        for k in data.keys():
            if not k.startswith("__"):
                signal = data[k].squeeze()
                break

        if signal is None:
            print(f"{file} signal not found")
            continue

        label_char = file[0]
        domain = get_domain(file, args.domain_mode)
        label = "normal" if label_char == "N" else "anomaly"

        save_path = os.path.join(args.save_dir, domain, label)
        os.makedirs(save_path, exist_ok=True)

        for i in range(0, len(signal) - args.window_size, args.window_size):
            segment = signal[i:i + args.window_size]

            segment = (segment - segment.min()) / (segment.max() - segment.min() + 1e-8)
            segment = 2 * segment - 1
            segment = segment.reshape(1, -1)

            gaf_img = gaf.fit_transform(segment)[0]

            # FIX (only changed line vs make_gadf.py): symmetric [-1,1]->[0,255]
            # remap instead of the wrap-around-prone `(gaf_img*255).astype(uint8)`.
            img = np.clip(np.round((gaf_img + 1) / 2 * 255), 0, 255).astype(np.uint8)
            img = Image.fromarray(img)

            save_name = f"{file[:-4]}_{i}.png"
            img.save(os.path.join(save_path, save_name))

        print(f"processed: {file} -> domain {domain}")

    print("\nGADF preprocessing (fixed) finished.")


if __name__ == "__main__":
    main()
