import os

def check_sizes():
    for root, dirs, files in os.walk('DataSets'):
        for f in files:
            path = os.path.join(root, f)
            print(f"{path}: {os.path.getsize(path) / 1024 / 1024:.2f} MB")
            
    for root, dirs, files in os.walk('models'):
        for f in files:
            path = os.path.join(root, f)
            print(f"{path}: {os.path.getsize(path) / 1024 / 1024:.2f} MB")

check_sizes()
