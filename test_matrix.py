import ctypes
import numpy as np
import time

# Load shared library
lib = ctypes.cdll.LoadLibrary("./libmatrix.so")

# Define argument types (PDF style: 1D contiguous float32 arrays)
lib.gpu_matrix_multiply.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int
]
lib.gpu_matrix_multiply.restype = None  # void

def run_once(N: int):
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    C = np.zeros((N, N), dtype=np.float32)

    t0 = time.time()
    lib.gpu_matrix_multiply(A.ravel(), B.ravel(), C.ravel(), N)
    t1 = time.time()

    ms = (t1 - t0) * 1000.0
    print(f"Python call to CUDA library completed in {ms:.3f} ms (N={N})")
    return ms

if __name__ == "__main__":
    for N in [512, 1024, 2048]:
        # warm-up
        run_once(N)

        # measured runs (3 times)
        times = [run_once(N) for _ in range(3)]
        print(f"Avg (N={N}): {sum(times)/len(times):.3f} ms\n")
