import sys
import os
import time

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")

start_time = time.time()
print("Attempting to import tensorflow...")

try:
    print("Importing DeepFace...")
    from deepface import DeepFace
    print("Successfully imported DeepFace.")
    import tensorflow as tf
    print(f"Tensorflow Version: {tf.__version__}")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

print(f"Detailed import took {time.time() - start_time:.2f} seconds")
