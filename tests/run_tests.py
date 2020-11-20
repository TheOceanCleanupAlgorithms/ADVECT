import glob
import os

tests = glob.glob(f'./**/*_test.py', recursive=True)

for test in tests:
    print(f"Running test {test}")
    os.system(f"python {test}")
