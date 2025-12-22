import cv2
import os
import time
import numpy as np
import multiprocessing
import concurrent.futures
import argparse

# --- 1. Image Filters [cite: 23] ---

def apply_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(image):
    return cv2.GaussianBlur(image, (3, 3), 0)

def apply_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.hypot(sobelx, sobely).astype(np.uint8)

def apply_sharpening(image):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def apply_brightness(image, value=30):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

# --- 2. Worker Function ---

def process_single_image(file_path):
    try:
        # Read image
        img = cv2.imread(file_path)
        if img is None:
            return f"Error: Could not read {file_path}"

        # Apply filters
        apply_grayscale(img)
        apply_gaussian_blur(img)
        apply_edge_detection(img)
        apply_sharpening(img)
        apply_brightness(img)
        
        return None # Success
    except Exception as e:
        return f"Exception on {file_path}: {str(e)}"

# --- 3. Parallel Implementations [cite: 37] ---

def run_multiprocessing(image_files, num_workers):
    start_time = time.time()
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(process_single_image, image_files)
    return time.time() - start_time

def run_concurrent_futures(image_files, num_workers):
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(process_single_image, image_files)) 
    return time.time() - start_time

# --- 4. Main Benchmark Driver ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Directory containing images")
    args = parser.parse_args()

    # --- 1. Find Images ---
    print(f"Scanning directory: {args.dir}")
    valid_exts = {".jpg", ".jpeg", ".png"}
    image_files = []

    for root, dirs, files in os.walk(args.dir):
        for file in files:
            # Skip macOS hidden files
            if file.startswith("._"):
                continue
            if os.path.splitext(file)[1].lower() in valid_exts:
                image_files.append(os.path.join(root, file))

    if not image_files:
        print("ERROR: No images found!")
        exit()

    # Limit to 500 images for consistency
    image_files = image_files[:500]
    print(f"Found {len(image_files)} images for testing.")

    # --- 2. Sanity Check (Test 1 Image) ---
    print("\n[Sanity Check] Testing first image...")
    test_result = process_single_image(image_files[0])
    if test_result is not None:
        print(f"FAILURE: {test_result}")
        exit()
    print("SUCCESS: Image processing logic is working.\n")

    # --- 3. Automated Benchmark Loop ---
    worker_counts = [1, 2, 4, 8]
    results = []
    
    # Store baseline times (T1) to calculate speedup
    t1_mp = 0 
    t1_cf = 0

    print(f"{'='*60}")
    print(f"Starting Benchmark Suite (Workers: {worker_counts})")
    print(f"{'='*60}")

    for n in worker_counts:
        print(f"Running tests with {n} workers...", end=" ", flush=True)
        
        # Run Multiprocessing
        t_mp = run_multiprocessing(image_files, n)
        
        # Run Concurrent Futures
        t_cf = run_concurrent_futures(image_files, n)
        
        print("Done.")

        # Calculate Metrics
        if n == 1:
            t1_mp = t_mp
            t1_cf = t_cf
            mp_speedup = 1.0
            cf_speedup = 1.0
            mp_eff = 1.0
            cf_eff = 1.0
        else:
            # Speedup = T1 / Tn
            mp_speedup = t1_mp / t_mp
            cf_speedup = t1_cf / t_cf
            
            # Efficiency = Speedup / n
            mp_eff = mp_speedup / n
            cf_eff = cf_speedup / n

        results.append({
            "workers": n,
            "mp_time": t_mp, "mp_su": mp_speedup, "mp_eff": mp_eff,
            "cf_time": t_cf, "cf_su": cf_speedup, "cf_eff": cf_eff
        })

    # --- 4. Generate Output Report ---
    print("\n\n" + "="*80)
    print("BENCHMARK REPORT (Copy this table to your assignment)")
    print("="*80)
    
    # Header
    print(f"{'Workers':<10} | {'MP Time(s)':<12} | {'MP Speedup':<12} | {'MP Eff':<10} || {'CF Time(s)':<12} | {'CF Speedup':<12} | {'CF Eff':<10}")
    print("-" * 95)

    # Rows
    for r in results:
        print(f"{r['workers']:<10} | {r['mp_time']:<12.4f} | {r['mp_su']:<12.2f} | {r['mp_eff']:<10.2f} || {r['cf_time']:<12.4f} | {r['cf_su']:<12.2f} | {r['cf_eff']:<10.2f}")
    
    print("-" * 95)
    print("Legend: MP = Multiprocessing, CF = Concurrent Futures, Eff = Efficiency")
    print("="*80 + "\n")