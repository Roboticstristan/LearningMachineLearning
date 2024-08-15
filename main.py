import numpy as np
np.random.seed(0)
X=[[1,2,3,2.5],
   [2.0,5.0,-1.0,2.0],
   [-1.5,2.7,3.3,-0.8]]

class LayerDense:
    def __init__(self,neuralImps,numOfNeurons):
        self.weights = 0.1* np.random.randn(neuralImps, numOfNeurons)
        self.biases= np.zeros((1,numOfNeurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

lay1 = LayerDense(len(X[0]),5)

lay2 = LayerDense(5,2)

lay1.forward(X)
z = lay2.forward(lay1.output)
print(lay2.output)