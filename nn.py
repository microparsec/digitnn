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
hidden_size = 100
output_size = len(labels)

# Training variables
batchsize = 50          # amount of samples to process each iteration
epochs = 100            # amount of times to look through the data set
alpha = 0.05            # amount of learning from one iteration
alphadecay = .8         # alpha decay factor when loss increases
alphacutoff = 0.0001    # stop converging if alpha drops below this threshold

softloss = 0.01         # smoothing factor for loss indication
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

def softmax(s):
    exps = np.exp(s - np.max(s, axis=-1,keepdims=True))
    return exps / np.sum(exps, axis=-1,keepdims=True)

def crossentropy(y_, y, e = 1e-12):
    return -np.sum(y_ * np.log(y + e)) 

def imagetostring(image):
    output = ""
    for y in range(0,imagesize):
        line = ""
        for x in range(0,imagesize):
            if image[imagesize * y + x] > 0.8: line += "#"
            elif image[imagesize * y + x] > 0.5: line += "$"
            elif image[imagesize * y + x] > 0.3: line += ";"
            else: line += " "
        output += line + "\n"
    return output

def batchtostring(batch):
    string = ""
    for y in batch:
        n = np.argmax(y)
        string += "%s %s\n" % (labels[n], str(np.round(y, 2)))
    return string


# Simple neural network with two hidden layers
class NeuralNetwork:

    

    def __init__(self, N = 2, H = 3, O = 1):
        self.N = N
        self.H = H
        self.O = O

        self.W1 = np.random.randn(N, H)
        self.B1 = np.zeros(H)
        self.W2 = np.random.randn(H, H)
        self.B2 = np.zeros(H)
        self.W3 = np.random.randn(H, O)
        self.B3 = np.zeros(O)
        
    def forward(self, X):
        self.z  = self.B1 + np.dot(X, self.W1)
        self.z2 = sigmoid(self.z)
        self.z3 = self.B2 + np.dot(self.z2, self.W2)
        self.z4 = sigmoid(self.z3)
        self.z5 = self.B3 + np.dot(self.z4, self.W3)
        return softmax(self.z5)

    def backward(self, X, y, o):
        #self.o_error = y - o
        self.o_derror = o - y

        self.z4_error = self.o_derror.dot(self.W3.T)
        self.z4_derror = self.z4_error * dsigmoid(self.z4)

        self.z2_error = self.z4_derror.dot(self.W2.T)
        self.z2_derror = self.z2_error * dsigmoid(self.z2)

        dW1 = X.T.dot(self.z2_derror)
        dB1 = np.sum(self.z2_derror, axis=0)
        dW2 = self.z2.T.dot(self.z4_derror)
        dB2 = np.sum(self.z4_derror, axis=0)
        dW3 = self.z4.T.dot(self.o_derror)
        dB3 = np.sum(self.o_derror, axis=0)
        return dW1, dB1, dW2, dB2, dW3, dB3


    def train(self, X, y, batchsize, alpha = 1):
        o = self.forward(X)
        dW1, dB1, dW2, dB2, dW3, dB3 = self.backward(X, y, o)

        self.W1 += -alpha * dW1
        self.B1 += -alpha * dB1
        self.W2 += -alpha * dW2
        self.B2 += -alpha * dB2
        self.W3 += -alpha * dW3
        self.B3 += -alpha * dB3

        return o

# Start the program

NN = NeuralNetwork(imagelength, hidden_size, len(labels))

# calculate starting loss value
forward = NN.forward(X[0:batchsize])
loss = crossentropy(y[0:batchsize],forward) / batchsize

starttime = time.time()

# TRAIN
error = False
for e in range(0, epochs):


    try:
        epoch_startloss = loss
        print('\rEpoch %i Loss: %.5f Alpha: %.5f ... ' % (e, loss, alpha), end="", flush = True)

        # shuffle the dataset
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)

        X = X[indices]
        y = y[indices]

        for i in range(0, int(trainingdatalength / batchsize)):
            batchstart = i * batchsize
            batchend = (i + 1) * batchsize

            forward = NN.train(X[batchstart:batchend], y[batchstart:batchend], batchsize, alpha)
            loss = crossentropy(y[batchstart:batchend], forward) / batchsize * softloss + (1-softloss) * loss

        #alpha transition
        if loss > epoch_startloss:
            alpha *= alphadecay
            if alpha <= alphacutoff:
                break

    except KeyboardInterrupt:
        error = True
        print(' aborted')
        break


if not error:
    print(" complete")
print("Trained %i epochs on %i samples in %.2f seconds \n" % (e + 1, trainingdatalength, time.time() - starttime))


print("Validating...\r", end="", flush="True")
errors = 0
show = True
for i in range(0,validationdatalength):
    prediction = NN.forward(vX[i:i+1])

    #       predicted               actual
    if np.argmax(prediction) != np.argmax(vy[i]):
        if show and not error:
            print(imagetostring(vX[i]))
            print("Actual: %s, Prediction: %s" % (labels[np.argmax(vy[i])], batchtostring(prediction)))
            print("Press 'q' to skip validation errors")
            if input() == "q":
                show = False

        errors += 1

print("Validation: %i out of %i - %.2f%%" % (validationdatalength - errors, validationdatalength, (100.0 * (validationdatalength - errors) / validationdatalength)) )

# Sampler

once = True
while((enable_sampler or once) and not error):
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

