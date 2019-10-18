import warnings
warnings.filterwarnings("ignore")
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import numpy as np

class pretrained():
    def __init__(self):
        self.maxlen = 40  # The window size
        self.charslen = 63
        self.char_indices = {'\t': 0, '\n': 1, ' ': 2, '!': 3, '"': 4, '&': 5, "'": 6, '(': 7, ')': 8, '*': 9, ',': 10, '-': 11, '.': 12, '/': 13, '0': 14, '1': 15, '2': 16, '3': 17, '4': 18, '5': 19, '6': 20, '7': 21, '8': 22, '9': 23, ':': 24, '=': 25, '?': 26, '[': 27, ']': 28, '_': 29, 'a': 30, 'b': 31, 'c': 32, 'd': 33, 'e': 34, 'f': 35, 'g': 36, 'h': 37, 'i': 38, 'j': 39, 'k': 40, 'l': 41, 'm': 42, 'n': 43, 'o': 44, 'p': 45, 'q': 46, 'r': 47, 's': 48, 't': 49, 'u': 50, 'v': 51, 'w': 52, 'x': 53, 'y': 54, 'z': 55, '|': 56, '\x7f': 57, 'à': 58, 'é': 59, '‘': 60, '’': 61, '…': 62}
        self.indices_char = {0: '\t', 1: '\n', 2: ' ', 3: '!', 4: '"', 5: '&', 6: "'", 7: '(', 8: ')', 9: '*', 10: ',', 11: '-', 12: '.', 13: '/', 14: '0', 15: '1', 16: '2', 17: '3', 18: '4', 19: '5', 20: '6', 21: '7', 22: '8', 23: '9', 24: ':', 25: '=', 26: '?', 27: '[', 28: ']', 29: '_', 30: 'a', 31: 'b', 32: 'c', 33: 'd', 34: 'e', 35: 'f', 36: 'g', 37: 'h', 38: 'i', 39: 'j', 40: 'k', 41: 'l', 42: 'm', 43: 'n', 44: 'o', 45: 'p', 46: 'q', 47: 'r', 48: 's', 49: 't', 50: 'u', 51: 'v', 52: 'w', 53: 'x', 54: 'y', 55: 'z', 56: '|', 57: '\x7f', 58: 'à', 59: 'é', 60: '‘', 61: '’', 62: '…'}

    def loadingmodel(self):
        self.model = Sequential()
        self.model.add(LSTM(128, input_shape=(self.maxlen, self.charslen)))
        self.model.add(Dense(self.charslen))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
        self.model.load_weights("onehot-modelweights.h5")

    def sample(init,preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def runmodel(self,sentence):
        variance = 0.4
        generated = ''
        original = sentence
        window = sentence
        self.loadingmodel()
        for i in range(400):
            x = np.zeros((1, self.maxlen, self.charslen))
            for t, char in enumerate(window):
                x[0, t, self.char_indices[char]] = 1.

            preds = self.model.predict(x, verbose=0)[0]
            next_index = self.sample(preds, variance)
            next_char = self.indices_char[next_index]

            generated += next_char
            window = window[1:] + next_char

        print(original + generated)
        print("As you can see there is ")

if __name__ == '__main__':
    start = pretrained()
    start.runmodel("shayi da kar de dhadka hai \n dil mein ho") # Accept 40 Characters only..





