import os
import argparse

import numpy as np
from scipy.fft import fft2, ifft2
import cv2

def open_image(image_fn: str) -> np.ndarray:
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    img: np.ndarray = cv2.imread(image_fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    print(f"Read image data type of {img.dtype}")
    if img.dtype == np.uint8 or img.dtype == np.uint16:
        img = img.astype(np.float32) / np.iinfo(img.dtype).max
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def show_image(image: np.ndarray):
    m, n, c = image.shape
    image = image.astype(np.float32)
    if c == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("image", image)
    cv2.waitKey(0)

def write_image(image_fn: str, img: np.ndarray):
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    h,w,c = img.shape
    if c < 3:
        needed_channels = 3 - c
        img = np.concatenate([img, np.zeros((h, w, needed_channels))], axis=2)
    # Reverse first axis.
    img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_fn, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        type=str,
        help="Image you'd like to open.",
    )
    args = parser.parse_args()

    img = open_image(args.filename)
    h, w, c = img.shape

    freqs = np.zeros_like(img).astype(np.complex64)
    for i in range(c):
        chan = img[:, :, i]
        freq = fft2(chan)
        print(freq.shape)
        freqs[:, :, i] = freq

    # TODO: replace this with something that better reflectst he desired MTF curve.
    kernel = cv2.getGaussianKernel(ksize=100, sigma=0)
    kernel = kernel @ kernel.T
    print("kernel shape: ", kernel.shape)
    filter_freq = np.expand_dims(fft2(kernel, s=(h, w)), 2) # shape (h, w, 1) so we can broadcast it to (h, w, 3)
    freqs *= filter_freq


    out_img = np.zeros_like(img)
    for i in range(c):
        freq = freqs[:, :, i]
        chan = ifft2(freq)
        out_img[:, :, i] = chan

    show_image(img)
    show_image(out_img)
