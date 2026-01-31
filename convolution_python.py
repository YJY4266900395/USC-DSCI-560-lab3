# convolution_python.py - Python wrapper for CUDA convolution library
import ctypes
import numpy as np
import time
import sys

# Load the shared library
lib = ctypes.CDLL("./libconvolution.so")

# Define argument types for gpu_convolution
lib.gpu_convolution.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags='C_CONTIGUOUS'),  # input
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags='C_CONTIGUOUS'),  # output
    ctypes.c_int,  # M
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),  # kernel
    ctypes.c_int   # N
]

# Define argument types for all helpers (input, output, M)
for func_name in ['gpu_sobel_x', 'gpu_sobel_y', 'gpu_laplacian', 
                  'gpu_sharpen', 'gpu_box_blur', 'gpu_gaussian', 'gpu_box_blur_7x7']:
    func = getattr(lib, func_name)
    func.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int
    ]


def load_raw(filename, M):
    """Load a .raw image file"""
    img = np.fromfile(filename, dtype=np.uint8)
    return img.astype(np.uint32)


def save_raw(filename, img):
    """Save image to .raw file"""
    img.astype(np.uint8).tofile(filename)


def convolution_cuda(input_img, kernel_name, M):
    """
    Apply convolution using CUDA library
    kernel_name: 'sobel_x', 'sobel_y', 'laplacian', 'sharpen', 'box', 'gaussian' (5*5), 'box7' (7*7)
    """
    output_img = np.zeros(M * M, dtype=np.uint32)
    
    if kernel_name == 'sobel_x':
        lib.gpu_sobel_x(input_img, output_img, M)
    elif kernel_name == 'sobel_y':
        lib.gpu_sobel_y(input_img, output_img, M)
    elif kernel_name == 'laplacian':
        lib.gpu_laplacian(input_img, output_img, M)
    elif kernel_name == 'sharpen':
        lib.gpu_sharpen(input_img, output_img, M)
    elif kernel_name == 'box':
        lib.gpu_box_blur(input_img, output_img, M)
    elif kernel_name == 'gaussian':
        lib.gpu_gaussian(input_img, output_img, M)
    elif kernel_name == 'box7':
        lib.gpu_box_blur_7x7(input_img, output_img, M)
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")
    
    return output_img


def main():   
    if len(sys.argv) < 5:
        print("Usage: python3 convolution_python.py <input.raw> <output.raw> <M> <kernel>")
        print("Kernels: sobel_x, sobel_y, laplacian, sharpen, box, gaussian, box7")
        print("\nExample:")
        print("  python3 convolution_python.py input.raw output.raw 512 sobel_x")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    M = int(sys.argv[3])
    kernel_name = sys.argv[4]
    
    print("=== Python CUDA Convolution ===")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"Image size: {M} x {M}")
    print(f"Kernel: {kernel_name}")
    
    # Load image
    input_img = load_raw(input_file, M)
    
    # Run convolution with timing
    # warm-up
    output_img = convolution_cuda(input_img, kernel_name, M)
    print("Processing...")
    start = time.time()
    output_img = convolution_cuda(input_img, kernel_name, M)
    end = time.time()
    
    elapsed = end - start
    print(f"Python calls CUDA time after warm-up: {elapsed:.6f} seconds ({elapsed*1000:.3f} ms)")

    times = []
    for i in range(5):
        start = time.time()
        output_img = convolution_cuda(input_img, kernel_name, M)
        end = time.time()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    print(f"Average time (5 runs): {avg_time:.6f} seconds ({avg_time*1000:.3f} ms)")
    
    # Save result
    save_raw(output_file, output_img)
    print(f"Output saved to: {output_file}")
    print("Done!")


if __name__ == "__main__":
    main()