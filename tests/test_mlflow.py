import os
import sys
import tempfile

from yumbox.mlflow import run_all_configs


# Test 1: Simple error that shows a clean traceback
def test_simple_error():
    """
    Create a simple Python script that raises an error with a clean traceback.
    This tests how run_all_configs handles standard errors.
    """
    # Create a temporary script that will generate an error
    error_script = """
import sys
import time

def main():
    print("Starting test script")
    time.sleep(0.5)  # Short delay to test real-time output
    print("About to raise an error")
    # This will generate a clean traceback
    raise ValueError("This is a test error with a multi-line\\nmessage that should\\nbe preserved intact")

if __name__ == "__main__":
    main()
    """

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(error_script)
        script_path = f.name

    # Create a fake config file
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write("# Test config")
        config_path = f.name

    try:
        # Create a configs directory with our test config
        configs_dir = tempfile.mkdtemp()
        config_target = os.path.join(configs_dir, os.path.basename(config_path))
        os.rename(config_path, config_target)

        # Run the test
        print("\n=== Testing Simple Error Handling ===")
        run_all_configs(
            configs_dir=configs_dir,
            configs_list=[config_target],
            mode="list",
            executable=sys.executable,
            script=script_path,
            config_arg="",
            extra_args=None,
        )

    finally:
        # Clean up
        if os.path.exists(script_path):
            os.unlink(script_path)
        if os.path.exists(config_target):
            os.unlink(config_target)
        if os.path.exists(configs_dir):
            os.rmdir(configs_dir)


# Test 2: Multiprocessing hanging error (like what happens with FAISS)
def test_multiprocessing_error():
    """
    Create a script that simulates the FAISS pickling error that hangs.
    This tests how run_all_configs handles subprocess hanging.
    """
    # Create a temporary script that will generate a multiprocessing error
    mp_error_script = """
import numpy as np
import faiss
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

# Create sample data
dimension = 128
num_vectors = 100
num_queries = 10

# Generate random data
vectors = np.random.random((num_vectors, dimension)).astype('float32')
queries = np.random.random((num_queries, dimension)).astype('float32')

# Create FAISS index
index = faiss.IndexFlatIP(dimension)
index.add(vectors)

def process_batch(batch_data, search_func, k):
    \"\"\"Process a batch with just the search function\"\"\"
    batch, idx = batch_data
    # This will fail with pickling error
    distances, indices = search_func(batch, k)
    return distances, indices

def main():
    queries_len = len(queries)
    batch_size = 20
    batches = [(queries[i:i+batch_size], i) for i in range(0, queries_len, batch_size)]
    
    # Get the search function from the index
    search_func = index.search
    
    with mp.Pool(processes=2) as pool:
        process_func = partial(process_batch, search_func=search_func, k=5)
        try:
            print("Starting multiprocessing...")
            results = list(
                tqdm(
                    pool.imap(process_func, batches),
                    total=len(batches),
                    desc="Processing batches",
                )
            )
            print("Successfully processed all batches!")
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")
            raise

if __name__ == "__main__":
    main()
    """

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(mp_error_script)
        script_path = f.name

    # Create a fake config file
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write("# Test config")
        config_path = f.name

    try:
        # Create a configs directory with our test config
        configs_dir = tempfile.mkdtemp()
        config_target = os.path.join(configs_dir, os.path.basename(config_path))
        os.rename(config_path, config_target)

        # Run the test
        print("\n=== Testing Multiprocessing Error Handling ===")
        run_all_configs(
            configs_dir=configs_dir,
            configs_list=[config_target],
            mode="list",
            executable=sys.executable,
            script=script_path,
            config_arg="",
            extra_args=None,
        )

    finally:
        # Clean up
        if os.path.exists(script_path):
            os.unlink(script_path)
        if os.path.exists(config_target):
            os.unlink(config_target)
        if os.path.exists(configs_dir):
            os.rmdir(configs_dir)


if __name__ == "__main__":
    # Run both tests
    test_simple_error()
    test_multiprocessing_error()
