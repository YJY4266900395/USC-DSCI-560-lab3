# image_utils.py - convert between PNG/JPG and RAW formats, and create test images
import numpy as np
from PIL import Image
import sys
import os
import matplotlib.pyplot as plt

def png_to_raw(input_path, output_path, size=None):
    """PNG/JPG --> RAW"""
    img = Image.open(input_path).convert('L')  # go black-white
    
    if size:
        img = img.resize((size, size))
    
    # make sure it's square
    w, h = img.size
    if w != h:
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))
    
    arr = np.array(img, dtype=np.uint8)
    arr.tofile(output_path)
    print(f"Saved: {output_path} ({arr.shape[0]}x{arr.shape[1]})")

def raw_to_png(input_path, output_path, size):
    """RAW --> PNG"""
    arr = np.fromfile(input_path, dtype=np.uint8)
    arr = arr.reshape((size, size))
    img = Image.fromarray(arr, mode='L')
    img.save(output_path)
    print(f"Saved: {output_path}")

def create_test_image(output_path, size=512):
    """generate a test image with patterns"""
    arr = np.zeros((size, size), dtype=np.uint8)
    
    # left-top: grid
    for i in range(size//2):
        for j in range(size//2):
            if ((i//32) + (j//32)) % 2 == 0:
                arr[i, j] = 250
            else:
                arr[i, j] = 50
    
    # right-top: horizontal gradient
    for i in range(size//2):
        for j in range(size//2, size):
            arr[i, j] = int((j - size//2) * 255 / (size//2))
    
    # left-bottom: vertical stripes
    for i in range(size//2, size):
        for j in range(size//2):
            arr[i, j] = 255 if (j % 64 < 32) else 0
    
    # right-bottom: circle
    cy, cx = 3*size//4, 3*size//4
    r = size//8
    for i in range(size//2, size):
        for j in range(size//2, size):
            if (i-cy)**2 + (j-cx)**2 < r**2:
                arr[i, j] = 255
            else:
                arr[i, j] = 50
    
    arr.tofile(output_path)
    print(f"Created test image: {output_path} ({size}x{size})")

def graph(input_path, output_path, size=512):
    """Display the RAW images as graph using matplotlib"""

    input_img = np.fromfile(input_path, dtype=np.uint8).reshape(size, size)
    output_img = np.fromfile(output_path, dtype=np.uint8).reshape(size, size)

    # show in a row
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(input_img, cmap='gray')
    ax1.set_title('Input')
    ax2.imshow(output_img, cmap='gray')
    ax2.set_title('Output')
    plt.show()

def print_usage():
    print("Usage:")
    print("  python image_utils.py png2raw <input.png> <output.raw> [size]")
    print("  python image_utils.py raw2png <input.raw> <output.png> <size>")
    print("  python image_utils.py test <output.raw> [size]")
    print("  python image_utils.py graph <image1.raw> <image2.raw> [size]")
    print("\nExamples:")
    print("  python image_utils.py test input.raw 512")
    print("  python image_utils.py png2raw photo.jpg input.raw 512")
    print("  python image_utils.py raw2png output.raw result.png 512")
    print("  python image_utils.py graph input.raw output.raw 256")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "png2raw":
        if len(sys.argv) < 4:
            print_usage()
            sys.exit(1)
        size = int(sys.argv[4]) if len(sys.argv) > 4 else None
        png_to_raw(sys.argv[2], sys.argv[3], size)
    
    elif cmd == "raw2png":
        if len(sys.argv) < 5:
            print_usage()
            sys.exit(1)
        raw_to_png(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    
    elif cmd == "test":
        if len(sys.argv) < 3:
            print_usage()
            sys.exit(1)
        size = int(sys.argv[3]) if len(sys.argv) > 3 else 512
        create_test_image(sys.argv[2], size)

    elif cmd == "graph":
        if len(sys.argv) < 4:
            print_usage()
            sys.exit(1)
        size = int(sys.argv[4]) if len(sys.argv) > 4 else 512
        graph(sys.argv[2], sys.argv[3], size)
    
    else:
        print(f"Unknown command: {cmd}")
        print_usage()
