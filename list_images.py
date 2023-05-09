import os

# Getting the current work directory (cwd)
thisdir = os.getcwd()

PATH = os.path.join(thisdir,'radiologist_test_images')

# r=root, d=directories, f = files
for r, d, f in os.walk(PATH):
    for file in f:
        if file.endswith(".png"):
            print(file)