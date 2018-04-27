import numpy as np
import random
import sys
import gzip
import imageio
import time


# ------- GLOBAL VARIABLES
# Training data
imagesize = 28
imagelength = imagesize * imagesize

labels = ["0","1","2","3","4","5","6","7","8","9"]

Xfile = "images.idx3-ubyte.gz"
yfile = "labels.idx1-ubyte.gz"
trainingdatalength = 60000

# Validation data
vXfile = "val.images.idx3-ubyte.gz"
vyfile = "val.labels.idx1-ubyte.gz"
validationdatalength = 10000

# sampler
sampleimage = "image.png"
enable_sampler = False

# network
input_size = imagelength
hidden_size = 50
output_size = len(labels)

# Training variables
batchsize = 10          # amount of samples to process each iteration
iterations = 10000
alpha = 0.1             # amount of learning from one iteration

softloss = 0.001        # smoothing factor for loss indication
outputinterval = 23     # iterations per terminal update

# TRAINING DATA SETUP
print("Initializing training and validation data ...", end="", flush=True)

with gzip.open(yfile) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(trainingdatalength)
    y = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    y = y.reshape(trainingdatalength,1)
    tempy = []
    for y_i in y:
        temp = [0.0] * len(labels)
        temp[int(y_i)] = 1.0
        tempy += temp
    y = np.array(tempy)
    y = y.reshape(trainingdatalength, len(labels))


with gzip.open(Xfile) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(imagelength * trainingdatalength)
    X = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    X = X / 255.0
    X = X.reshape(trainingdatalength, imagelength)

with gzip.open(vyfile) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(validationdatalength)
    vy = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    vy = vy.reshape(validationdatalength,1)
    tempvy = []
    for vy_i in vy:
        temp = [0.0] * len(labels)
        temp[int(vy_i)] = 1.0
        tempvy += temp
    vy = np.array(tempvy)
    vy = vy.reshape(validationdatalength, len(labels))


with gzip.open(vXfile) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(imagelength * validationdatalength)
    vX = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    vX = vX / 255.0
    vX = vX.reshape(validationdatalength, imagelength)

print(" complete")

def sigmoid(s):
    return 1 / (1 + np.exp(-s))

def dsigmoid(s):
    return s * (1 - s)

def imagetostring(image):
    output = ""
    for y in range(0,imagesize):
        line = ""
        for x in range(0,imagesize):
            if image[imagesize * y + x] > 0.5: line += "#"
            else: line += " "
        output += line + "\n"
    return output

def batchtostring(batch):
    string = ""
    for y in batch:
        n = np.argmax(y)
        string += str(n) +" " + str(np.round(y, 2)) + "\n"
    return string


# Simple neural network with two hidden layers
class NeuralNetwork:

    def __init__(self, N = 2, H = 3, O = 1):
        self.N = N
        self.H = H
        self.O = O

        self.W1 = np.random.randn(N, H)
        self.W2 = np.random.randn(H, H)
        self.W3 = np.random.randn(H, O)
        
    def forward(self, X):
        self.z = np.dot(X, self.W1)
        self.z2 = sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        self.z4 = sigmoid(self.z3)
        self.z5 = np.dot(self.z4, self.W3)

        return sigmoid(self.z5)

    def backward(self, X, y, o):
        self.o_error = y - o
        self.o_derror = self.o_error * dsigmoid(o)

        self.z4_error = self.o_error.dot(self.W3.T)
        self.z4_derror = self.z4_error * dsigmoid(self.z4)

        self.z2_error = self.z4_derror.dot(self.W2.T)
        self.z2_derror = self.z2_error * dsigmoid(self.z2)

        dW1 = X.T.dot(self.z2_derror)
        dW2 = self.z2.T.dot(self.z4_derror)
        dW3 = self.z4.T.dot(self.o_derror)
        return dW1, dW2, dW3


    def train(self, X, y, alpha = 1):
        o = self.forward(X)
        dW1, dW2, dW3 = self.backward(X, y, o)

        self.W1 += alpha * dW1
        self.W2 += alpha * dW2
        self.W3 += alpha * dW3


# Start the program

NN = NeuralNetwork(imagelength, hidden_size, len(labels))

# calculate starting loss value
forward = NN.forward(X[0:batchsize])
loss = np.mean(np.square(y[0:batchsize] - forward))

starttime = time.time()

# TRAIN
for i in range(0, iterations):

    batchstart = np.random.randint(0, trainingdatalength - batchsize)
    batchend = batchstart + batchsize

    forward = NN.forward(X[batchstart:batchend])
    loss = np.mean(np.square(y[batchstart:batchend] - forward)) * softloss + (1-softloss) * loss

    if i % outputinterval == 0 or i == iterations - 1:
        print('\rIteration %i Loss: %.5f ...' % (i + 1, loss), end="", flush = True)

    # alpha transition
    # if i % 100000 == 0:
    #     alpha /= 2

    NN.train(X[batchstart:batchend], y[batchstart:batchend], alpha)
print(" complete")
print("Trained batches of %i iterations on %i samples in %.2f seconds \n" % (batchsize, iterations, time.time() - starttime))


print("Validating...\r", end="", flush="True")
errors = 0
for i in range(0,validationdatalength):
    prediction = NN.forward(vX[i])

    #       predicted               actual
    if np.argmax(prediction) != np.argmax(vy[i]):
        errors += 1

print("Validation: %i out of %i - %.2f%%" % (validationdatalength - errors, validationdatalength, (100.0 * (validationdatalength - errors) / validationdatalength)) )

# Sampler

once = True
while(enable_sampler or once):
    once = False

    image = imageio.imread(sampleimage).reshape(imagesize, imagesize, 4)
    tmp = [[(float(x[0]) + float(x[1]) + float(x[2])) / 3.0 for x in line] for line in image] 
    image = (np.array(tmp) / 255.0).reshape(1, imagelength)

    print(imagetostring(image[0]))
    prediction = NN.forward(image)
    output = np.argmax(prediction[0])
    print("Prediction: %s" % batchtostring(prediction))

    try:
        if not enable_sampler or input().strip() == "exit":
            enable_sampler = False
    except:
        print("Please type 'exit' to close the sampler")
        enable_sampler = False

