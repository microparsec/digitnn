import sys
import gzip
import random
import numpy as np
from collections import defaultdict

# THIS SCRIPT CONVERTS A CSV FILE WITH IMAGES TO AN IDX FILE LIKE THE MNIST DATASET
# THE CSV FILE MUST HAVE THE FOLLOWING LAYOUT:
# int(label), 784 * int(pixelvalue)
# So you get rows starting with the label number, and then all the pixel values

handwritingfile = "data/handwritten_data_785.csv"           # Search for this on Google. It's on Kaggle

# New file names
trainingimagesfile = "data/letters.images.idx3-ubyte.gz"
traininglabelsfile = "data/letters.labels.idx1-ubyte.gz"
validationimagesfile = "data/val.letters.images.idx3-ubyte.gz"
validationlabelsfile = "data/val.letters.labels.idx1-ubyte.gz"

def writeimagesfile(filename, images):
    with gzip.open(filename, 'wb') as bytestream:
        bytestream.write(0x00000803.to_bytes(4, "big"))
        bytestream.write(len(images).to_bytes(4, "big"))
        bytestream.write((28).to_bytes(4, "big"))
        bytestream.write((28).to_bytes(4, "big"))
        for image in images:
            bytestream.write(bytes(image))

def writelabelsfile(filename, labels):
    with gzip.open(filename, 'wb') as bytestream:
        bytestream.write(0x00000801.to_bytes(4, "big"))
        bytestream.write(len(labels).to_bytes(4, "big"))
        for label in labels:
            bytestream.write(label.to_bytes(1, "big", signed=False))

with open(handwritingfile, 'r') as csv:
    hlabels = []
    himages = []
    
    i = 0
    print("Reading file ... ", end="", flush=True)
    for line in csv:
        line = line.split(',')
        hlabels.append(int(line[0]))
        himage = list(map(int, line[1:]))
        himages.append(himage)
        i += 1
    print("complete", flush=True)


    print("Shuffling ... ", end="", flush=True)
    mixer = list(zip(himages, hlabels))
    random.shuffle(mixer)
    himages, hlabels = zip(*mixer)
    print("complete", flush=True)

    print("Splitting ... ", end="", flush=True)
    pivot = int(round(len(hlabels) / 6.0))
    trainingimages = himages[pivot:]
    traininglabels = hlabels[pivot:]
    validationimages = himages[0:pivot]
    validationlabels = hlabels[0:pivot]
    if not len(traininglabels) + len(validationlabels) == len(hlabels):
        print(" failed, sizes don't add up")
    statistic = defaultdict(int)
    for x in hlabels:
        statistic[x] += 1
    values = list(statistic.values())
    print("Mean: %.2f Stddev: %.2f" % (np.mean(values), np.std(values)))
    print("complete", flush=True)


    print("Outputting ... ", end="", flush=True)
    writeimagesfile(trainingimagesfile, trainingimages)
    writelabelsfile(traininglabelsfile, traininglabels)
    writeimagesfile(validationimagesfile, validationimages)
    writelabelsfile(validationlabelsfile, validationlabels)
    print("complete", flush=True)
