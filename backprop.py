"""
My first perceptron class
Author: Jack Bosco
1/30/23
"""
import numpy as np
import os

def printTable(board, leftHeaders = False):
        # Print the top border
        if leftHeaders:
            print('  ' + "-"*11 * len(board[0]) + "-")
        else:
            print("-"*11 * len(board[0]) + "-")
        # Iterate through the board and print Q if we have a queen there
        for row in range(len(board)):
            if leftHeaders:
                print(row, '||', end='')
            else:
                print("| ", end="")
            for col in range(len(board[0])):
                # Print a blank space for no queen
                print("{:^8}".format(str(round(board[row][col],5))) + " | ", end="")
            print("\n" + "-"*11 * len(board[0]) + "-")
            
def progressBar(progress=1.0, size=50, message='', full = '█', empty = ' '):
    """prints a progress bar of size characters. progress is float value 0.0 to 1.0"""
    return '{4:>3}%|{0:{1}<{2}s}|{3:<{2}}'.format(full*int(progress*size), empty, size, message,  int(progress*100))

def section(name):
    return '\n{0:=^100s}'.format(name)

class Perceptron():
    #===========================================================Part 1===========================================================
    def __init__(self, h, i, o):

        # wIH ← (n+1) x h matrix of small random +/- values close to 0
        self.wih = self.new_weights(i, h)

        # wHO ← (h+1) x m matrix of small random +/- values close to 0
        self.who = self.new_weights(h, o)

    def new_weights(self, n1, n2):
        return np.random.random((n1+1, n2)) * 1 - 0.5
    
    def __str__(self):
        return "A perceptron matrix with {0} inputs and {1} outputs".format(self.nInputs, self.nOutputs)
    
    def test(self, I):
        I1 = np.append(I, [1])
        Hnet = np.dot(I1, self.wih)
        # print('wih', self.wih.size)
        H = self.squash(Hnet)
        H1 = np.append(H, [1])
        # print(H1.size)
        Onet = np.dot(H1, self.who)
        return self.squash(Onet)

        
    def squash(self, net): 
        # I think this is a sigmoidal function
        # oj = f (net j) = 1 / (1 + e^-netj )
        return 1 / (1 + np.exp(-net))
    
    def dSquash(self, net): 
       # first derivative of the squashing function (sigmoidal function)
       return self.squash(net) * (1-self.squash(net))
    
    def train(self, I, T, neps=1000, eta=0.05, mew=0):
        #previous weight changes for momentum
        DwHOprev, DwIHprev = 0, 0
        
        # for i ←1 to 1000 do // 1000 is a convenient number 
        for i in range(neps): # just a nice-looking thing

            if i % (neps//100) == 0:
                print(progressBar(progress=(i+1)/neps), end='\r')

            # ∆wIH ← (n+1) x h matrix of zeros // weight changes
            DwIH = np.zeros(self.wih.shape)

            # ∆wHO ← (h+1) x m matrix of zeros // weight changes 
            DwHO = np.zeros(self.who.shape)

            # The number of patterns
            p = len(I)

            # for each pattern 1 ≤ j ≤ p do
            for j in range(p):

                I1 = np.append(I[j], [1])

                # Hnet ←[Ij ,1] * wIH  // append bias, multiply by weights
                Hnet = np.dot(I1, self.wih)

                # H ← f (Hnet )  // squashing function
                H = self.squash(Hnet)

                H1 = np.append(H, [1])

                # Onet ←[H ,1] * wHO  // append bias, multiply by weights
                Onet = np.dot(H1, self.who)

                # O ← f (Onet )  // squashing function for sigmoidal, i guess.
                O = self.squash(Onet)

                # δO ← (Tj – O) * f ' (Onet )  // f ' is first derivative of squashing function
                dO = (T[j] - O) * self.dSquash(Onet)

                # δH ← δO  * wT HO * f ' (Hnet )  // this is back-prop
                eH = np.dot(dO, self.who.T)[:-1]
                dH = eH * self.dSquash(Hnet)

                # what i had:          dH = (dO*wHO)[:-1] * self.dSquash(Hnet)
                #∆wIH  ← ∆wIH + [Ij ,1]  * δH // learning
                DwIH += np.outer(I1, dH)

                # ∆wHO ← ∆wHO + [H ,1] * δO
                DwHO += np.outer(H1, dO)

            # wIH   ← wIH + h * ∆wIH      / p//  h = small positive constant, a.k.a. learning rate
            self.wih += (eta * DwIH + mew * DwIHprev) / p
            self.who += (eta * DwHO + mew * DwHOprev) / p 
            DwIHprev = DwIH
            DwHOprev = DwHO
        print(progressBar(message='Done!'))
    
    def save(self, ih='inputHidden.npy', ho='hiddenOutput.npy'):
        np.save(ih, self.wih, allow_pickle=True)
        np.save(ho, self.who, allow_pickle=True)
        print('files saved to', os.getcwd())
    
    def load(self, ih='inputHidden.npy', ho='hiddenOutput.npy'):
        self.wih = np.load(ih, allow_pickle=True)
        self.who = np.load(ho, allow_pickle=True)
        


    

#=========================================================== main ==========================================================
def main():
    pass
    
if __name__ == '__main__':
    main()
