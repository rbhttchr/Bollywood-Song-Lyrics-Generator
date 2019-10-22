import warnings
warnings.filterwarnings("ignore")
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import numpy as np
import pandas as pd


class OneHotWayEncoding():

    def dataimport(self):
        path = 'dataset.csv'
        self.df = pd.read_csv(path)
        print(self.df.head(5))

    def tocorpus(self):
        self.text = self.df['Lyrics'].str.cat(sep='\n').lower()
        # Output the length of the Corpus
        print('corpus length : ', len(self.text))

        # Create a sorted list of the Characters
        self.chars = sorted(list(set(self.text)))
        print('Total Chars : ', len(self.chars))

        # Reducing the length of Corpus
        self.text = self.text[:900999]
        print('Truncated Corpus Length : ', len(self.text))

    def featurelabeldataset(self):
        # Create a dictionary where given a character, you can look up the index and vice versa
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

        # cut the text in semi-redundant sequences of maxlen characters
        self.maxlen = 40  # The window size
        self.step = 3  # The steps between the windows
        sentences = []
        next_chars = []

        # Step through the text via 3 characters at a time, taking a sequence of 40 bytes at a time.
        # There will be lots ofo overlap
        for i in range(0, len(self.text) - self.maxlen, self.step):
            sentences.append(self.text[i: i + self.maxlen])  # range from current index i for max length characters
            next_chars.append(self.text[i + self.maxlen])  # the next character after that

        self.sentences = np.array(sentences)
        self.next_chars = np.array(next_chars)
        print('Number of sequences:', len(sentences))

        data = {'Sentences': sentences, 'Next Char': next_chars}

        # Create DataFrame
        newdf = pd.DataFrame(data)

        # Print the output.
        print(newdf)

    def getdata(self,sentences, next_chars):
        X = np.zeros((len(sentences), self.maxlen, len(self.chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(self.chars)), dtype=np.bool)
        length = len(sentences)
        index = 0
        for i in range(len(sentences)):
            sentence = sentences[i]
            for t, char in enumerate(sentence):
                X[i, t, self.char_indices[char]] = 1
            y[i, self.char_indices[next_chars[i]]] = 1
        return X, y

    def buildmodel(self):
        # build the model: a single LSTM
        print('Build model...')
        self.model = Sequential()
        self.model.add(LSTM(128, input_shape=(self.maxlen, len(self.chars))))
        self.model.add(Dense(len(self.chars)))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

        print("Compiling model complete...")

    def trainingmodel(self):
        self.dataimport()
        self.tocorpus()
        self.featurelabeldataset()
        self.buildmodel()
        X, y = self.getdata(self.sentences, self.next_chars)
        # The training
        print('Training...')
        # Use this if they all fit into memory
        history = self.model.fit(X, y, batch_size=128, epochs=30, verbose=1)
        # Save the model
        self.model.save('onehot-modelweights.h5')

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
        for i in range(400):
            x = np.zeros((1, self.maxlen, len(self.chars)))
            for t, char in enumerate(window):
                x[0, t, self.char_indices[char]] = 1.

            preds = self.model.predict(x, verbose=0)[0]
            next_index = self.sample(preds, variance)
            next_char = self.indices_char[next_index]

            generated += next_char
            window = window[1:] + next_char

        print(original + generated)

if __name__ == '__main__':
    start = OneHotWayEncoding()
    start.trainingmodel()
    start.runmodel("shayi da kar de dhadka hai \n dil mein ho")