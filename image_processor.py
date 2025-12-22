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

# --- 1: Grayscale Conversion ---
def apply_grayscale(image):
    """
    Converts a BGR image to Grayscale.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        
    Returns:
        numpy.ndarray: Grayscale image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --- 2: Gaussian Blur ---
def apply_gaussian_blur(image):
    """
    Applies a Gaussian Blur filter to smooth the image and reduce noise.
    Uses a 3x3 kernel.
    
    Args:
        image (numpy.ndarray): Input image.
        
    Returns:
        numpy.ndarray: Blurred image.
    """
    return cv2.GaussianBlur(image, (3, 3), 0)

# --- 3: Sobel Filter ---
def apply_edge_detection(image):
    """
    Detects edges in the image using the Sobel operator.
    Calculates the gradient magnitude from Sobel X and Sobel Y derivatives.
    
    Args:
        image (numpy.ndarray): Input image.
        
    Returns:
        numpy.ndarray: Edge-detected image (gradient magnitude).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    # Combine the two gradients to find the magnitude
    return np.hypot(sobelx, sobely).astype(np.uint8)

# --- 4: Image Sharpening ---
def apply_sharpening(image):
    """
    Enhances the edges and details of an image using a sharpening kernel.
    
    Kernel used:
    [[ 0, -1,  0],
     [-1,  5, -1],
     [ 0, -1,  0]]
     
    Args:
        image (numpy.ndarray): Input image.
        
    Returns:
        numpy.ndarray: Sharpened image.
    """
    kernel = np.array([[0, -1, 0], 
                       [-1, 5,-1], 
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# --- 5: Brightness Adjustment ---
def apply_brightness(image, value=30):
    """
    Adjusts the brightness of the image by modifying the Value (V) channel 
    in the HSV color space.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        value (int): Amount to increase brightness (default: 30).
        
    Returns:
        numpy.ndarray: Brightness-adjusted image in BGR format.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Add constant value to the V channel
    v = cv2.add(v, value)
    
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


# =============================================================================
# SECTION 2: WORKER FUNCTION
# =============================================================================

def process_single_image(file_path):
    """
    Worker function executed by each parallel process. 
    Reads an image from disk and sequentially applies all 5 filters.
    
    Note: To focus benchmarks on CPU processing time, the modified images 
    are not written back to disk.
    
    Args:
        file_path (str): The full path to the image file.
        
    Returns:
        None if successful, or an error string if an exception occurs.
    """
    try:
        # Read image from disk
        img = cv2.imread(file_path)
        
        # Check if image loaded correctly
        if img is None:
            return f"Error: Could not read {file_path}"

        # Apply the processing pipeline sequentially
        # 1. Grayscale
        apply_grayscale(img)
        # 2. Gaussian Blur
        apply_gaussian_blur(img)
        # 3. Edge Detection
        apply_edge_detection(img)
        # 4. Sharpening
        apply_sharpening(img)
        # 5. Brightness
        apply_brightness(img)
        
        return None # Return None to indicate success
        
    except Exception as e:
        return f"Exception on {file_path}: {str(e)}"


# =============================================================================
# SECTION 3: PARALLEL IMPLEMENTATIONS
# =============================================================================

def run_multiprocessing(image_files, num_workers):
    """
    Executes the image processing tasks using the `multiprocessing` module.
    
    This approach uses a Pool of worker processes. The `map` function blocks 
    until all results are returned, ensuring synchronization.
    
    Args:
        image_files (list): List of file paths to process.
        num_workers (int): Number of parallel processes to spawn.
        
    Returns:
        float: Total execution time in seconds.
    """
    start_time = time.time()
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        # map() divides the 'image_files' list into chunks and sends them to workers
        pool.map(process_single_image, image_files)
        
    end_time = time.time()
    return end_time - start_time

def run_concurrent_futures(image_files, num_workers):
    """
    Executes the image processing tasks using the `concurrent.futures` module.
    
    This approach uses ProcessPoolExecutor, which provides a higher-level 
    abstraction over multiprocessing.
    
    Args:
        image_files (list): List of file paths to process.
        num_workers (int): Number of parallel processes to spawn.
        
    Returns:
        float: Total execution time in seconds.
    """
    start_time = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(process_single_image, image_files)) 
        
    end_time = time.time()
    return end_time - start_time


# =============================================================================
# SECTION 4: MAIN BENCHMARK DRIVER
# =============================================================================

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Parallel Image Processing Benchmark")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing input images")
    args = parser.parse_args()

    # --- Step 1: Discover Images ---
    print(f"Scanning directory: {args.dir}")
    valid_exts = {".jpg", ".jpeg", ".png"}
    image_files = []

    # Recursively walk through directory to find all valid images
    for root, dirs, files in os.walk(args.dir):
        for file in files:
            # Filter out hidden metadata files (e.g., macOS ._ files)
            if file.startswith("._"):
                continue
            
            if os.path.splitext(file)[1].lower() in valid_exts:
                image_files.append(os.path.join(root, file))

    if not image_files:
        print("ERROR: No images found! Check your directory path.")
        exit()

    # Query number of images found in the dataset
    print(f"Found {len(image_files)} images for benchmarking.")

    # --- Step 2: Sanity Check ---
    # Validate that the pipeline works on a single image before running the full test
    print("\n[Sanity Check] Testing processing logic on the first image...")
    test_result = process_single_image(image_files[0])
    
    if test_result is not None:
        print(f"FAILURE: The sanity check failed. Reason: {test_result}")
        exit()
    print("SUCCESS: Image processing pipeline is functional.\n")

    # --- Step 3: Automated Benchmark Loop ---
    # Test with 1, 2, 4, and 8 workers to analyze scalability
    worker_counts = [1, 2, 4, 8]
    results = []
    
    # Variables to store serial execution time (1 worker) for Speedup calculation
    t1_mp = 0 
    t1_cf = 0

    print(f"{'='*80}")
    print(f"Starting Benchmark Suite (Testing with {worker_counts} workers)")
    print(f"{'='*80}")

    for n in worker_counts:
        print(f"Running tests with {n} workers...", end=" ", flush=True)
        
        # 3a. Benchmark Multiprocessing
        t_mp = run_multiprocessing(image_files, n)
        
        # 3b. Benchmark Concurrent Futures
        t_cf = run_concurrent_futures(image_files, n)
        
        print(f"Done. (MP: {t_mp:.2f}s, CF: {t_cf:.2f}s)")

        # 3c. Calculate Performance Metrics
        if n == 1:
            # Baseline (Serial) Execution
            t1_mp = t_mp
            t1_cf = t_cf
            mp_speedup = 1.0
            cf_speedup = 1.0
            mp_eff = 1.0
            cf_eff = 1.0
        else:
            # Speedup = Time_Serial / Time_Parallel
            mp_speedup = t1_mp / t_mp
            cf_speedup = t1_cf / t_cf
            
            # Efficiency = Speedup / Number_of_Workers
            mp_eff = mp_speedup / n
            cf_eff = cf_speedup / n

        # Store results
        results.append({
            "workers": n,
            "mp_time": t_mp, "mp_su": mp_speedup, "mp_eff": mp_eff,
            "cf_time": t_cf, "cf_su": cf_speedup, "cf_eff": cf_eff
        })

    # --- Step 4: Generate Output Report ---
    print("\n\n" + "="*95)
    print("BENCHMARK REPORT (Copy this table to your technical report)")
    print("="*95)
    
    # Print Table Header
    print(f"{'Workers':<10} | {'MP Time(s)':<12} | {'MP Speedup':<12} | {'MP Eff':<10} || {'CF Time(s)':<12} | {'CF Speedup':<12} | {'CF Eff':<10}")
    print("-" * 95)

    # Print Table Rows
    for r in results:
        print(f"{r['workers']:<10} | {r['mp_time']:<12.4f} | {r['mp_su']:<12.2f} | {r['mp_eff']:<10.2f} || {r['cf_time']:<12.4f} | {r['cf_su']:<12.2f} | {r['cf_eff']:<10.2f}")
    
    print("-" * 95)
    print("Legend: MP = Multiprocessing, CF = Concurrent Futures, Eff = Efficiency")
    print("="*95 + "\n")