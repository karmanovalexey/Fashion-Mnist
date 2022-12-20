import os

def mymkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)