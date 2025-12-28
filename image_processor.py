"""
========================================================================
        CST435 Assignment 2: Parallel Image Processing System
========================================================================

* This script implements a benchmarking tool to compare the performance
  of two Python parallel programming paradigms:

  1. Multiprocessing (via multiprocessing.Pool)
  2. Concurrent Futures (via concurrent.futures.ProcessPoolExecutor)

* The system processes a collection of images by applying a pipeline 
  of five computationally intensive filters manually implemented using
  NumPy operations:

  - Grayscale Conversion (Luminance Formula)
  - Gaussian Blur (Convolution with 3x3 Kernel)
  - Edge Detection (Sobel Operator)
  - Image Sharpening (Convolution)
  - Brightness Adjustment (Pixel-wise Addition)

"""

import cv2
import os
import time
import numpy as np
import multiprocessing
import concurrent.futures
import argparse

# =============================================================================
# SECTION 1: IMAGE PROCESSING FILTERS
# =============================================================================

# Helper function for 2D convolution
def convolve2d(image, kernel):
    """
    Helper function to apply a 2D convolution operation on a grayscale image 
    using a specific kernel.
    
    This manually implements the sliding window mechanism:
    1. Pads the image to handle borders.
    2. Iterates over every pixel.
    3. Computes the weighted sum of the neighborhood defined by the kernel.
    
    Args:
        image (numpy.ndarray): Grayscale input image.
        kernel (numpy.ndarray): 2D filter matrix (e.g., 3x3 Gaussian).
        
    Returns:
        numpy.ndarray: Convolved image.
    """
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # Pad the image with edge values to maintain dimensions
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    output = np.zeros_like(image, dtype=np.float32)

    # Iterate over the image dimensions
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract the region of interest (ROI)
            region = padded[i:i+kh, j:j+kw]
            # Apply element-wise multiplication and summation
            output[i, j] = np.sum(region * kernel)

    return np.clip(output, 0, 255).astype(np.uint8)


# Filter 1: Grayscale Conversion
def apply_grayscale(image):
    """
    Converts a BGR color image to a single-channel Grayscale image using the 
    standard Luminance formula.
    
    Formula: Gray = 0.299*R + 0.587*G + 0.114*B
    
    Args:
        image (numpy.ndarray): Input image in BGR format (Height, Width, 3).
        
    Returns:
        numpy.ndarray: Grayscale image (Height, Width) of type uint8.
    """
    # Extract B, G, R channels
    b = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    r = image[:, :, 2].astype(np.float32)

    # Apply luminance weights
    gray = 0.114 * b + 0.587 * g + 0.299 * r
    
    # Clip values to 0-255 range and convert back to 8-bit integer
    return np.clip(gray, 0, 255).astype(np.uint8)


# Filter 2: Gaussian Blur
def apply_gaussian_blur(image):
    """
    Applies a Gaussian Blur filter to smooth the image and reduce noise.
    Uses a manual 3x3 Gaussian kernel approximation.
    
    Kernel:
    [[1, 2, 1],
     [2, 4, 2],
     [1, 2, 1]] / 16.0
    
    Args:
        image (numpy.ndarray): Input BGR image.
        
    Returns:
        numpy.ndarray: Blurred grayscale image.
    """
    # Convert to grayscale
    gray = apply_grayscale(image)

    # Define 3x3 Gaussian kernel
    kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=np.float32) / 16.0

    return convolve2d(gray, kernel)


# Filter 3: Edge Detection
def apply_edge_detection(image):
    """
    Detects edges in the image using the Sobel operator.
    Calculates the gradient magnitude by convolving with Sobel-X and Sobel-Y kernels.
    
    Formula: Magnitude = sqrt(Gx^2 + Gy^2)
    
    Args:
        image (numpy.ndarray): Input BGR image.
        
    Returns:
        numpy.ndarray: Edge magnitude map.
    """
    gray = apply_grayscale(image)

    # Sobel kernel for horizontal changes
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    # Sobel kernel for vertical changes
    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)

    # Apply convolutions
    gx = convolve2d(gray, sobel_x)
    gy = convolve2d(gray, sobel_y)

    # Calculate gradient magnitude
    magnitude = np.sqrt(gx.astype(np.float32)**2 + gy.astype(np.float32)**2)
    
    return np.clip(magnitude, 0, 255).astype(np.uint8)


# Filter 4: Image Sharpening
def apply_sharpening(image):
    """
    Enhances the edges and details of an image using a sharpening convolution kernel.
    
    Kernel:
    [[ 0, -1,  0],
     [-1,  5, -1],
     [ 0, -1,  0]]
     
    Args:
        image (numpy.ndarray): Input BGR image.
        
    Returns:
        numpy.ndarray: Sharpened grayscale image.
    """
    gray = apply_grayscale(image)

    # Laplacian-based sharpening kernel
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)

    return convolve2d(gray, kernel)


# Filter 5: Brightness Adjustment
def apply_brightness(image, value=30):
    """
    Adjusts the brightness of the image by adding a constant value to every pixel.
    
    Args:
        image (numpy.ndarray): Input BGR image.
        value (int): Brightness offset (default: 30).
        
    Returns:
        numpy.ndarray: Brightness-adjusted BGR image.
    """
    # Convert to int16 to prevent overflow during addition
    result = image.astype(np.int16) + value
    
    # Clip values to valid [0, 255] range and cast back to uint8
    return np.clip(result, 0, 255).astype(np.uint8)


# =============================================================================
# SECTION 2: WORKER FUNCTION
# =============================================================================

def process_single_image(file_path):
    """
    Worker function executed by each parallel process.
    Reads an image from disk and sequentially applies all 5 manual filters.
    
    Args:
        file_path (str): The full path to the image file.
        
    Returns:
        None if successful, or an error string if an exception occurs.
    """
    try:
        # Read image from disk
        img = cv2.imread(file_path)

        # Validate that image loaded correctly
        if img is None:
            return f"Error: Could not read {file_path}"

        # Apply the manual processing pipeline
        apply_grayscale(img)
        apply_gaussian_blur(img)
        apply_edge_detection(img)
        apply_sharpening(img)
        apply_brightness(img)

        return None # Indicate success

    except Exception as e:
        return f"Exception on {file_path}: {str(e)}"


# =============================================================================
# SECTION 3: PARALLEL IMPLEMENTATIONS
# =============================================================================

def run_multiprocessing(image_files, num_workers):
    """
    Executes the image processing tasks using the `multiprocessing` module.
    
    Mechanism:
    - Creates a Pool of worker processes.
    - Uses pool.map() to distribute the list of files across workers.
    - Each process runs in its own memory space, bypassing the Global Interpreter Lock (GIL).
    
    Args:
        image_files (list): List of file paths to process.
        num_workers (int): Number of parallel processes to spawn.
        
    Returns:
        float: Total execution time in seconds.
    """
    start_time = time.time()

    with multiprocessing.Pool(processes=num_workers) as pool:
        # map() blocks until all tasks are complete
        pool.map(process_single_image, image_files)

    return time.time() - start_time


def run_concurrent_futures(image_files, num_workers):
    """
    Executes the image processing tasks using the `concurrent.futures` module.
    
    Mechanism:
    - Uses ProcessPoolExecutor as a high-level abstraction over multiprocessing.
    - Returns futures (placeholders for eventual results).
    
    Args:
        image_files (list): List of file paths to process.
        num_workers (int): Number of parallel processes to spawn.
        
    Returns:
        float: Total execution time in seconds.
    """
    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # list() is forced here to ensure the iterator completes execution 
        # before stopping the timer
        list(executor.map(process_single_image, image_files))

    return time.time() - start_time


# =============================================================================
# SECTION 4: MAIN BENCHMARK DRIVER
# =============================================================================

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Parallel Image Processing Benchmark")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing images")
    args = parser.parse_args()

    # --- Step 1: Discover Images ---
    valid_exts = {".jpg", ".jpeg", ".png"}
    image_files = []

    # Recursively scan directory
    for root, _, files in os.walk(args.dir):
        for file in files:
            # Filter out macOS hidden metadata files
            if file.startswith("._"):
                continue
            if os.path.splitext(file)[1].lower() in valid_exts:
                image_files.append(os.path.join(root, file))

    if not image_files:
        print("Error: No images found. Please check the directory path.")
        exit()
        
    print(f"Found {len(image_files)} images for benchmarking.")

    # --- Step 2: Sanity Check ---
    print("\n[Sanity Check] Testing pipeline on the first image...")
    check_result = process_single_image(image_files[0])
    if check_result is not None:
        print(f"Sanity check failed: {check_result}")
        exit()
    print("Sanity check passed. Pipeline is functional.\n")

    # --- Step 3: Automated Benchmark Loop ---
    worker_counts = [1, 2, 4, 8]
    results = []

    # Variables to store baseline (serial) times for speedup calculation
    t1_mp = 0 
    t1_cf = 0

    print(f"{'='*95}")
    print(f"Starting Benchmark Suite (Workers: {worker_counts})")
    print(f"{'='*95}")

    for n in worker_counts:
        print(f"Running tests with {n} workers...", end=" ", flush=True)

        # Run benchmarks
        t_mp = run_multiprocessing(image_files, n)
        t_cf = run_concurrent_futures(image_files, n)

        print(f"Done. (MP: {t_mp:.2f}s, CF: {t_cf:.2f}s)")

        # Calculate Metrics
        if n == 1:
            t1_mp, t1_cf = t_mp, t_cf
            mp_su = cf_su = mp_eff = cf_eff = 1.0
        else:
            # Speedup = Time_Serial / Time_Parallel
            mp_su = t1_mp / t_mp
            cf_su = t1_cf / t_cf
            
            # Efficiency = Speedup / N
            mp_eff = mp_su / n
            cf_eff = cf_su / n

        results.append((n, t_mp, mp_su, mp_eff, t_cf, cf_su, cf_eff))

    # --- Step 4: Generate Report ---
    print("\n\n" + "="*95)
    print("BENCHMARK RESULTS REPORT")
    print("="*95)
    print(f"{'Workers':<10} | {'MP Time(s)':<12} | {'MP Speedup':<10} | {'MP Eff':<8} || {'CF Time(s)':<12} | {'CF Speedup':<10} | {'CF Eff':<8}")
    print("-" * 95)

    for r in results:
        print(f"{r[0]:<10} | {r[1]:<12.4f} | {r[2]:<10.2f} | {r[3]:<8.2f} || {r[4]:<12.4f} | {r[5]:<10.2f} | {r[6]:<8.2f}")

    print("="*95 + "\n")
    print("Legend: MP = Multiprocessing, CF = Concurrent Futures, SU = Speedup, Eff = Efficiency" + "\n\n")