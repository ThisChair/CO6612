import numpy as np

def svd_solve(a,b):
    """ Solves the systme Ax = b using svd"""
    u,s,v = np.linalg.svd(a)
    c = np.dot(u.T,b)
    y = np.array([w/z for (w,z) in (zip(c,s))])
    x = np.dot(v.T,y)
    return x



class RBF:
    def __init__(self, inputsize, centernum, sigma=1.0):
        """
        Radial Basis Function neural network.
        inputsize: Dimension of input data.
        centernum: Dimension of hidden layer.
        """
        self.inputsize = inputsize
        self.centernum = centernum
        self.sigma = sigma
        self.centers = np.zeros(inputsize)
        self.w = np.zeros(centernum)
        
    def activation(self,c,d):
        """
        Gaussian activation function.
        c: Center position.
        d: Input data.
        """
        return np.exp(-(1 / (2 * self.sigma**2)) * np.linalg.norm(c-d)**2)

    def interpolationMatrix(self, inputs):
        """
        Applies activation to the input.
        """
        phi = np.array([[self.activation(c,d) for c in self.centers] for d in inputs])
        g0 = np.array([[self.activation(c,d) for c in self.centers] for d in self.centers])
        return phi, g0

    def randomCenters(self, inputs):
        self.centers = inputs[np.random.choice(len(inputs), self.centernum)]
    
    def train(self, inputs, target, reg):
        self.randomCenters(inputs)
        phi, g0 = self.interpolationMatrix(inputs)
        a = np.dot(phi.T,phi) + (reg * g0)
        #self.w = np.dot(np.linalg.pinv(phi), target)
        #self.w = svd_solve(phi,target)
        self.w = svd_solve(a,np.dot(phi.T,target))

    def predict(self,x):
        y = [self.activation(c,x) for c in self.centers]
        return sum(y * self.w)

