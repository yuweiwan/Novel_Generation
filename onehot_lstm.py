import keras
import sys
import numpy as np
from keras import layers


def delete_brackets(s):
    # delete all decorations and brackets
    p_left = [index for (index, value) in enumerate(s) if value == '(']
    p_right = [index for (index, value) in enumerate(s) if value == ')']
    sen = s
    for i in range(len(p_left)):
        sen = sen.replace(s[p_left[i]: p_right[i] + 1], '')
    words = sen.split(' ')
    while '' in words:
        words.remove('')
    return words


def sample(preds, temperature=1.0):
    if not isinstance(temperature, float) and not isinstance(temperature, int):
        print("temperature must be a number")
        raise TypeError

    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def write(model, temperature, word_num, begin_sentence):
    gg = begin_sentence[:30]
    print(gg, end='/// ')
    for _ in range(word_num):
        sampled = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(gg):
            sampled[0, t, char_indices[char]] = 1.0

        preds = model.predict(sampled, verbose=0)[0]
        if temperature is None:
            next_word = chars[np.argmax(preds)]
        else:
            next_index = sample(preds, temperature)
            next_word = chars[next_index]

        gg += next_word
        gg = gg[1:]
        sys.stdout.write(next_word)
        sys.stdout.flush()


text = open('Cthulhu.txt').read()

# if using "whole.txt", activate following line; Else, need not.
# text = delete_brackets(text)

maxlen = 30
sentences = []
next_chars = []
begin_sentence = text[200: 300]
print(begin_sentence[:30])
# recording a moving window (size=maxlen) and following char
for i in range(0, len(text) - maxlen):
    sentences.append(text[i:i + maxlen])
    next_chars.append(text[i + maxlen])
print('total sentence count', len(text), len(sentences))

chars = sorted(list(set(text)))  # all unique chars, in other words,"dictionary".
char_indices = dict((char, chars.index(char)) for char in chars)

x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)  # sentence count, sequence length, dictionary length
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)  # sentence count, dictionary length
# one-hot
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1.0
        y[i, char_indices[next_chars[i]]] = 1.0

# print(np.round((sys.getsizeof(x) / 1024 / 1024 / 1024), 2), "GB")
# print(x.shape, y.shape)

model = keras.models.Sequential()
model.add(layers.LSTM(256, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))

optimizer = keras.optimizers.RMSprop(lr=1e-3)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.fit(x, y, epochs=100, batch_size=1024, verbose=2)

begin_sentence = text[200: 300]
print(begin_sentence[:30])

write(model, None, 1000, begin_sentence)
