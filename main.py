import numpy as np
import nnfs
import math

nnfs.init()


def createData(points,classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

class LayerDense:
    def __init__(self,neuralImps,numOfNeurons):
        self.weights = 0.1* np.random.randn(neuralImps, numOfNeurons)
        self.biases= np.zeros((1,numOfNeurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class activationRectifiedLinearUnit:
    def foward(self,inputs):
        self.output = np.maximum(0,inputs)

class activationSoftMax:
    def foward(self,inputs):
        expVal = np.exp(inputs - np.max(inputs, axis =1, keepdims = True))
        probabilities = expVal/np.sum(expVal, axis = 1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, predictedVal, y):
        sampleLosses = self.forward(predictedVal,y)
        data_loss = np.mean(sampleLosses)
        return data_loss
    
class categoricalCrossEntrophy(Loss):
    def forward(self, yPred, oneHotEncoded):
        samples = len(yPred)
        yPredClip = np.clip(yPred,1e-7, 1-1e-7)

        if len(oneHotEncoded.shape) ==1:
            CorrectConfidences = yPredClip[range(samples),oneHotEncoded]
        
        elif len(oneHotEncoded.shape) == 2:
            CorrectConfidences = np.sum(yPredClip*oneHotEncoded, axis = 1)
        
        loss = -np.log(CorrectConfidences)
        return loss
    
X,y = createData(100, 3)

dense1 = LayerDense(2,3)
activation1 =activationRectifiedLinearUnit()

dense2 = LayerDense(3,3)
activation2 = activationSoftMax()

dense1.forward(X)
activation1.foward(dense1.output)
dense2.forward(activation1.output)
activation2.foward(dense2.output)
#print(activation2.output)

lossFunc = categoricalCrossEntrophy()
loss = lossFunc.calculate(activation2.output,y)
print("loss: " + str(loss))

