#Import Libraries
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 20 #(Maximum length is 20 for both languages)

#################### Data Processing ############################

# Create dictionaries
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
        self.total = 0

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)


    def addWord(self, word):

        if word.lower() not in self.word2index:
            self.word2index[word.lower()] = self.n_words
            self.word2count[word.lower()] = 1
            self.index2word[self.n_words] = word.lower()
            self.n_words += 1
            self.total+=1
        else:
            self.word2count[word.lower()] += 1
            self.total+=1

# Turn a Unicode string to plain ASCII, thanks to

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
# Lowercase, trim, and remove non-letter characters

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.,':-;""!?])#@*&%<>/", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

##################### Read DataSets #########################


def readLangs(lang1, lang2,reverse):
    #Read data files
    print("Reading lines...")
    europarl_es = open('data/es-en/europarl-%s.es' %(lang1), encoding='utf-8').read().split('\n')
    europarl_en = open('data/es-en/europarl-%s.en' %(lang2), encoding='utf-8').read().split('\n')
    raw_data = {'Spanish' : [normalizeString(line) for line in europarl_es], 'English': [normalizeString(line) for line in europarl_en]}
    # Create a dataframe
    df = pd.DataFrame(raw_data, columns=["Spanish", "English"])
    # Make a list of sentences
    pairs = df.values.tolist()

    #Checking if Spanish-English or English-Spanish translator 
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


# Filtering out the sentences whose length is longer than 20
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH
        
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


##################### Details about the dataset ######################
def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)

    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    print("Total counted words:")
    print(input_lang.name, input_lang.total)
    print(output_lang.name, output_lang.total)
    print('')
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData('spanish', 'english', False)


################# Split dataSet ##################
# Train 80%, Valid 10%, Test 10%

from sklearn.model_selection import train_test_split
train_pairs, dev = train_test_split(pairs, test_size=0.2)
valid_pairs, test_pairs = train_test_split(dev, test_size=0.5)

print("Number of training pairs", len(train_pairs))
print("Number of validation pairs", len(valid_pairs))
print("Number of testing pairs", len(test_pairs))
print('')



################# Encoder ##################
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size


        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


################# Decoder ##################

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size,dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        #self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout_p = nn.Dropout(dropout_p)

    def forward(self, input, hidden,encoder_outputs):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output) # relu activation function
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

################# Get Index and Tensor ##################

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word.lower()] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


################## Train data #############################

teacher_forcing_ratio = 0.5
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


######################## Get the time #########################
import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

################ Call the train() Function #################

def trainIters(encoder, decoder, n_iters, print_every, learning_rate=0.01):
    start = time.time()
    print_loss_total = 0  # Reset every print_every

    for epoch in range(epochs):
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
        training_pairs = [tensorsFromPair(random.choice(train_pairs))
                          for i in range(n_iters)]
        criterion = nn.NLLLoss()
    
        for iter in range(1, n_iters + 1):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]
    
            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            
    
            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg))
  
########################### Evaluate the Model #########################

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden
        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words

import sacrebleu
from sacremoses import MosesDetokenizer
md = MosesDetokenizer(lang='en')

##################### Evaluate and Get the BLEU Score ###################
def evaluateBLEU(encoder, decoder,dataset,n=20):
    ref=[]
    pred=[]
    for i in range(n):
        pair = random.choice(dataset)
        output_words = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join((output_words))        
        ref.append(pair[1])
        pred.append(output_sentence)
    ref=[ref]
    bleu = sacrebleu.corpus_bleu(pred, ref)
    print(bleu.score)

############### Print 20 translated sentence reandomly ###############    
def printRandomly(encoder, decoder, n=20):
    for i in range(n):
        pair = random.choice(train_pairs)
        print('Source =', pair[0])
        print('Target =', pair[1])
        output_words = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join((output_words))        
        print('Translated = ', output_sentence)
        print('')   

################ Model and Output ##################
print("This is a seq2seq NMT without attention")        
epochs = 2
hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder1 = DecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
trainIters(encoder1, decoder1, (len(train_pairs)), print_every=(len(train_pairs)/20))
print('')
print("BLEUScore of validation Data: ")
evaluateBLEU(encoder1, decoder1, valid_pairs)
print("BLEUScore of Test Data: ")
evaluateBLEU(encoder1, decoder1, test_pairs)
print('')
print("Randomly printed 20 words: ")
print('')
printRandomly(encoder1, decoder1, n=20)    


