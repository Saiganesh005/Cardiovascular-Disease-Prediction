import os

def create_directories():

    os.makedirs("output/images", exist_ok=True)

    os.makedirs("models", exist_ok=True)
