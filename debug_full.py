import sys
import os
import time

print("Starting full import test...")
start = time.time()

print("Importing cv2...")
import cv2
print(f"Imported cv2. Time: {time.time()-start:.2f}s")

print("Importing numpy...")
import numpy as np
print(f"Imported numpy. Time: {time.time()-start:.2f}s")

print("Importing torch...")
import torch
print(f"Imported torch. Time: {time.time()-start:.2f}s")

print("Importing DeepFace...")
from deepface import DeepFace
print(f"Imported DeepFace. Time: {time.time()-start:.2f}s")

print("All imports successful.")
