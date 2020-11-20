import glob
import importlib
import os

from config import ROOT_DIR

tests = glob.glob(f'./**/*_test.py', recursive=True)

for test in tests:
    print(f"Running test {test}")
    os.system(f"python {test}")
