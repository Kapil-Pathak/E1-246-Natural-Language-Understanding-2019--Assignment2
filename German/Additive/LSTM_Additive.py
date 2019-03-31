import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import nltk
import random
import numpy as np
from collections import Counter, OrderedDict
import nltk
from copy import deepcopy
import os
import re
import unicodedata
flatten = lambda l: [item for sublist in l for item in sublist]

from torch.nn.utils.rnn import PackedSequence,pack_padded_sequence
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
random.seed(1024)

USE_CUDA = torch.cuda.is_available()
gpus = [0]
torch.cuda.set_device(gpus[0])

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex=0
    eindex=batch_size
    while eindex < len(train_data):
        batch = train_data[sindex: eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch

    if eindex >= len(train_data):
        batch = train_data[sindex:]
        yield batch

def pad_to_batch(batch, x_to_ix, y_to_ix):

    sorted_batch =  sorted(batch, key=lambda b:b[0].size(1), reverse=True) # sort by len
    x,y = list(zip(*sorted_batch))
    max_x = max([s.size(1) for s in x])
    max_y = max([s.size(1) for s in y])
    x_p, y_p = [], []
    for i in range(len(batch)):
        if x[i].size(1) < max_x:
            x_p.append(torch.cat([x[i], Variable(LongTensor([x_to_ix['<PAD>']] * (max_x - x[i].size(1)))).view(1, -1)], 1))
        else:
            x_p.append(x[i])
        if y[i].size(1) < max_y:
            y_p.append(torch.cat([y[i], Variable(LongTensor([y_to_ix['<PAD>']] * (max_y - y[i].size(1)))).view(1, -1)], 1))
        else:
            y_p.append(y[i])
    input_var = torch.cat(x_p)
    target_var = torch.cat(y_p)
    input_len = [list(map(lambda s: s ==0, t.data)).count(False) for t in input_var]
    target_len = [list(map(lambda s: s ==0, t.data)).count(False) for t in target_var]

    return input_var, target_var, input_len, target_len

def prepare_sequence(seq, to_index):
    idxs = list(map(lambda w: to_index[w] if to_index.get(w) is not None else to_index["<UNK>"], seq))
    return Variable(LongTensor(idxs))

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([,.!?])", r" \1 ", s)
    s = re.sub(r"[^a-zA-Z]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

corpus=open('/home/kapil/Additive/deu.txt', 'r').readlines()
corpus=corpus[:100000]
print(len(corpus))

MIN_LENGTH = 3
MAX_LENGTH = 25
i=0
X_r, y_r = [], [] # raw
X_t, y_t = [], []
X_tr, y_tr = [], []
for parallel in corpus:
    so,ta = parallel[:-1].split('\t')
    if so.strip() == "" or ta.strip() == "":
        continue

    normalized_so = normalize_string(so).split()
    normalized_ta = normalize_string(ta).split()

    if len(normalized_so) >= MIN_LENGTH and len(normalized_so) <= MAX_LENGTH and len(normalized_ta) >= MIN_LENGTH and len(normalized_ta) <= MAX_LENGTH:
        if i%9!=0:
            X_r.append(normalized_so)
            y_r.append(normalized_ta)
        else:
            X_t.append(normalized_so)
            y_t.append(normalized_ta)
    
    i=i+1

source_vocab = list(set(flatten(X_r)))
target_vocab = list(set(flatten(y_r)))
print(len(source_vocab), len(target_vocab))

source2index = {'<PAD>': 0, '<UNK>': 1, '<s>': 2, '</s>': 3}
for vo in source_vocab:
    if source2index.get(vo) is None:
        source2index[vo] = len(source2index)
index2source = {v:k for k, v in source2index.items()}

target2index = {'<PAD>': 0, '<UNK>': 1, '<s>': 2, '</s>': 3}
for vo in target_vocab:
    if target2index.get(vo) is None:
        target2index[vo] = len(target2index)
index2target = {v:k for k, v in target2index.items()}

X_p, y_p = [], []

for so, ta in zip(X_r, y_r):
    X_p.append(prepare_sequence(so + ['</s>'], source2index).view(1, -1))
    y_p.append(prepare_sequence(ta + ['</s>'], target2index).view(1, -1))
X_p1, y_p1 = [], []
for so, ta in zip(X_t, y_t):
    X_p1.append(prepare_sequence(so + ['</s>'], source2index).view(1, -1))
    y_p1.append(prepare_sequence(ta + ['</s>'], target2index).view(1, -1))

del source_vocab, target_vocab
train_data = list(zip(X_p, y_p))
print("Total Training Examples: ")
print(len(train_data))
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size,hidden_size, n_layers=1,bidirec=False):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, embedding_size)

        if bidirec:
            self.n_direction = 2
            #self.gru = nn.GRU(embedding_size, hidden_size, n_layers, batch_first=True, bidirectional=True)
            self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, batch_first=True, bidirectional=True)
        else:
            self.n_direction = 1
            #self.gru = nn.GRU(embedding_size, hidden_size, n_layers, batch_first=True)

    def init_hidden(self, inputs):
        hidden = Variable(torch.zeros(self.n_layers * self.n_direction, inputs.size(0), self.hidden_size))
        return hidden.cuda() if USE_CUDA else hidden

    def initContext(self, inputs):
        context = Variable(torch.zeros(self.n_layers *2, inputs.size(0), self.hidden_size))
        return context.cuda()

    def init_weight(self):
        self.embedding.weight = nn.init.xavier_uniform(self.embedding.weight)
        self.lstm.weight_hh_l0 = nn.init.xavier_uniform(self.lstm.weight_hh_l0)
        self.lstm.weight_ih_l0 = nn.init.xavier_uniform(self.lstm.weight_ih_l0)

    def forward(self, inputs, input_lengths):
        """
        inputs : B, T (LongTensor)
        input_lengths : real lengths of input batch (list)
        """
        hidden = self.init_hidden(inputs)
        context = self.initContext(inputs)
        embedded = self.embedding(inputs)
        packed = pack_padded_sequence(embedded, input_lengths, batch_first=True)
        #outputs, hidden = self.gru(packed, hidden)
        outputs,( hidden, context) = self.lstm(packed, (hidden,context))
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True) # unpack (back to padded)

        if self.n_layers > 1:
            if self.n_direction == 2:
                hidden = hidden[-2:]
                context=context[-2:]
            else:
                hidden = hidden[-1]

        return outputs, torch.cat([h for h in hidden], 1).unsqueeze(1), torch.cat([h for h in context], 1).unsqueeze(1)

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=1, dropout_p=0.1):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.tanh=nn.Tanh()
        # Define the layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.dropout = nn.Dropout(dropout_p)

        #self.gru = nn.GRU(embedding_size + hidden_size, hidden_size, n_layers, batch_first=True)
        self.lstm = nn.LSTM(embedding_size + hidden_size, hidden_size, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, input_size)
        self.attn_1= nn.Linear(self.hidden_size, self.hidden_size) # Attention
        self.attn_2 = nn.Linear(self.hidden_size, self.hidden_size)
    def init_hidden(self,inputs):
        hidden = Variable(torch.zeros(self.n_layers, inputs.size(0), self.hidden_size))
        return hidden.cuda() if USE_CUDA else hidden

    def initContext(self, inputs):
        context = Variable(torch.zeros(self.n_layers, inputs.size(0), self.hidden_size))
        return context.cuda()

    def init_weight(self):
        self.embedding.weight = nn.init.xavier_uniform(self.embedding.weight)
        self.lstm.weight_hh_l0 = nn.init.xavier_uniform(self.lstm.weight_hh_l0)
        self.lstm.weight_ih_l0 = nn.init.xavier_uniform(self.lstm.weight_ih_l0)
        self.linear.weight = nn.init.xavier_uniform(self.linear.weight)
        self.attn_1.weight = nn.init.xavier_uniform(self.attn_1.weight)
        self.attn_2.weight = nn.init.xavier_uniform(self.attn_2.weight)
#         self.attn.bias.data.fill_(0)

    def Multiplicative_Attention(self, hidden, encoder_outputs, encoder_maskings):
        """
        hidden : 1,B,D
        encoder_outputs : B,T,D
        encoder_maskings : B,T # ByteTensor
        """
        hidden = hidden[0].unsqueeze(2)  # (1,B,D) -> (B,D,1)

        batch_size = encoder_outputs.size(0) # B
        max_len = encoder_outputs.size(1) # T
        energies = self.attn(encoder_outputs.contiguous().view(batch_size * max_len, -1)) # B*T,D -> B*T,D
        energies = energies.view(batch_size,max_len, -1) # B,T,D
        #energies = self.tanh(energies)
        #energies = self.attn_v(energies)
        attn_energies = energies.bmm(hidden).squeeze(2) # B,T,D * B,D,1 --> B,T

#         if isinstance(encoder_maskings,torch.autograd.variable.Variable):
#             attn_energies = attn_energies.masked_fill(encoder_maskings,float('-inf'))#-1e12) # PAD masking

        alpha = F.softmax(attn_energies,1) # B,T
        alpha = alpha.unsqueeze(1) # B,1,T
        context = alpha.bmm(encoder_outputs) # B,1,T * B,T,D => B,1,D

        return context, alpha

    def Additive_Attention(self, hidden, encoder_outputs, encoder_maskings):
        """
        hidden : 1,B,D
        encoder_outputs : B,T,D
        encoder_maskings : B,T # ByteTensor
        """
        hidden = hidden[0].unsqueeze(2)  # (1,B,D) -> (B,D,1)

        batch_size = encoder_outputs.size(0) # B
        max_len = encoder_outputs.size(1) # T
        energies = self.attn_1(encoder_outputs.contiguous().view(batch_size * max_len, -1)) # B*T,D -> B*T,D
        energies = energies.view(batch_size,max_len, -1) # B,T,D
        energies = self.tanh(energies)
        energies = self.attn_2(energies)
        attn_energies = energies.bmm(hidden).squeeze(2) # B,T,D * B,D,1 --> B,T

#         if isinstance(encoder_maskings,torch.autograd.variable.Variable):
#             attn_energies = attn_energies.masked_fill(encoder_maskings,float('-inf'))#-1e12) # PAD masking

        alpha = F.softmax(attn_energies,1) # B,T
        alpha = alpha.unsqueeze(1) # B,1,T
        context = alpha.bmm(encoder_outputs) # B,1,T * B,T,D => B,1,D

        return context, alpha



    def forward(self, inputs, context, max_length, encoder_outputs, encoder_maskings=None, is_training=False):
        """
        inputs : B,1 (LongTensor, START SYMBOL)
        context : B,1,D (FloatTensor, Last encoder hidden state)
        max_length : int, max length to decode # for batch
        encoder_outputs : B,T,D
        encoder_maskings : B,T # ByteTensor
        is_training : bool, this is because adapt dropout only training step.
        """
        # Get the embedding of the current input word
        embedded = self.embedding(inputs)
        hidden = self.init_hidden(inputs)
        context_c = self.initContext(inputs)
        if is_training:
            embedded = self.dropout(embedded)

        decode = []
        # Apply GRU to the output so far
        for i in range(max_length):
            _, (hidden,context_c) = self.lstm(torch.cat((embedded, context), 2), (hidden,context_c))
            #_, hidden = self.gru(torch.cat((embedded, context), 2), hidden) # h_t = f(h_{t-1},y_{t-1},c)
            concated = torch.cat((hidden, context.transpose(0, 1)), 2) # y_t = g(h_t,y_{t-1},c)
            score = self.linear(concated.squeeze(0))
            softmaxed = F.log_softmax(score,1)
            decode.append(softmaxed)
            decoded = softmaxed.max(1)[1]
            embedded = self.embedding(decoded).unsqueeze(1) # y_{t-1}
            if is_training:
                embedded = self.dropout(embedded)

            # compute next context vector using attention
            context, alpha = self.Additive_Attention(hidden, encoder_outputs, encoder_maskings)

        #  column-wise concat, reshape!!
        scores = torch.cat(decode, 1)
        return scores.view(inputs.size(0) * max_length, -1)

    def decode(self, context, encoder_outputs):
        start_decode = Variable(LongTensor([[target2index['<s>']] * 1])).transpose(0, 1)
        embedded = self.embedding(start_decode)
        hidden = self.init_hidden(start_decode)
        context_c = self.initContext(start_decode)
        decodes = []
        attentions = []
        decoded = embedded
        max_len=25
        i=0
        while decoded.data.tolist()[0] != target2index['</s>']and i<max_len: # until </s>
            #_, hidden = self.gru(torch.cat((embedded, context), 2), hidden) # h_t = f(h_{t-1},y_{t-1},c)
            i=i+1
            #print(i,max_len)
            _, (hidden,_) = self.lstm(torch.cat((embedded, context), 2), (hidden,context_c))
            concated = torch.cat((hidden, context.transpose(0, 1)), 2) # y_t = g(h_t,y_{t-1},c)
            score = self.linear(concated.squeeze(0))
            softmaxed = F.log_softmax(score,1)
            decodes.append(softmaxed)
            decoded = softmaxed.max(1)[1]
            embedded = self.embedding(decoded).unsqueeze(1) # y_{t-1}
            #print(encoder_outputs.shape)
            #print(hidden.shape)
            context, alpha = self.Additive_Attention(hidden, encoder_outputs,None)
            attentions.append(alpha.squeeze(1))

        return torch.cat(decodes).max(1)[1], torch.cat(attentions)

EPOCH =20
BATCH_SIZE = 256
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 512
LR = 0.001
DECODER_LEARNING_RATIO = 5.0
RESCHEDULED = False

encoder = Encoder(len(source2index), EMBEDDING_SIZE, HIDDEN_SIZE, 3, True)
decoder = Decoder(len(target2index), EMBEDDING_SIZE, HIDDEN_SIZE * 2)
encoder.init_weight()
decoder.init_weight()

if USE_CUDA:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

loss_function = nn.CrossEntropyLoss(ignore_index=0)
enc_optimizer = optim.Adam(encoder.parameters(), lr=LR)
dec_optimizer = optim.Adam(decoder.parameters(), lr=LR * DECODER_LEARNING_RATIO)
f1=open("Loss.txt","w+")
from tqdm import tqdm_notebook as tqdm
torch.backends.cudnn.benchmark=True
for epoch in tqdm(range(EPOCH)):
    losses=[]
    for i, batch in enumerate(getBatch(BATCH_SIZE, train_data)):
        inputs, targets, input_lengths, target_lengths = pad_to_batch(batch, source2index, target2index)
        input_masks = torch.cat([Variable(ByteTensor(tuple(map(lambda s: s ==0, t.data)))) for t in inputs]).view(inputs.size(0), -1)
        start_decode = Variable(LongTensor([[target2index['<s>']] * targets.size(0)])).transpose(0, 1)
        encoder.zero_grad()
        decoder.zero_grad()
        output, hidden_c, context_c = encoder(inputs, input_lengths)

        preds = decoder(start_decode, hidden_c, targets.size(1), output, input_masks, True)

        loss = loss_function(preds, targets.view(-1))
        losses.append(loss.data.tolist())
        loss.backward()
        torch.nn.utils.clip_grad_norm(encoder.parameters(), 50.0) # gradient clipping
        torch.nn.utils.clip_grad_norm(decoder.parameters(), 50.0) # gradient clipping
        enc_optimizer.step()
        dec_optimizer.step()
        if i % 5==0:
            print("[%02d/%d] [%03d/%d] mean_loss : %0.2f" %(epoch, EPOCH, i, len(train_data)//BATCH_SIZE, np.mean(losses)))
            if ((len(train_data)//BATCH_SIZE)-i)<5:
                f1.write(str(np.mean(losses))+"\n")
            losses=[]

f1.close()


MODEL_SAVE_PATH="/home/kapil/Additive/encoder.pt"
#torch.save(decoder.state_dict(), MODEL_SAVE_PATH)
torch.save(encoder.state_dict(), MODEL_SAVE_PATH)
MODEL_SAVE_PATH1="/home/kapil/Additive/decoder.pt"
#torch.save(decoder.state_dict(), MODEL_SAVE_PATH)
torch.save(decoder.state_dict(), MODEL_SAVE_PATH1)


def show_attention(input_words, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_words, rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

#     show_plot_visdom()
    plt.show()
    plt.savefig('LSTM_Additive.png')
    plt.close()
test_data = list(zip(X_p1, y_p1))
target_reference=[]
hypothesis=[]
source_reference=[]

target_reference_t=[]
hypothesis_t=[]
source_reference_t=[]

print("Total Testing Examples:")
print(len(test_data))
print("Initializing testing......")
for i in range(10000):
    print(i)
    #test = random.choice(test_data)
    test = random.choice(train_data)
    input_ = test[0]
    truth = test[1]
    #print(input_)
    #print(len(input_))
    output, hidden, context_c = encoder(input_, [input_.size(1)])
    pred, attn = decoder.decode(hidden, output)

    input_ = [index2source[i] for i in input_.data.tolist()[0]]
    pred = [index2target[i] for i in pred.data.tolist()]
    source_reference.append(input_)
    target_reference.append(truth)
    hypothesis.append(pred)

for i in range(10000):
    print(i)
    test = random.choice(test_data)
    #test = random.choice(train_data)
    input_ = test[0]
    truth = test[1]
    #print(input_)
    #print(len(input_))
    output, hidden, context_c = encoder(input_, [input_.size(1)])
    pred, attn = decoder.decode(hidden, output)

    input_ = [index2source[i] for i in input_.data.tolist()[0]]
    pred = [index2target[i] for i in pred.data.tolist()]
    source_reference_t.append(input_)
    target_reference_t.append(truth)
    hypothesis_t.append(pred)



bleuscore_unigram=[]
bleuscore_4_gram=[]
for i in range(10000):
    bleuscore_unigram.append(nltk.translate.bleu_score.sentence_bleu([target_reference[i]], hypothesis[i],weights=(1,0,0,0)))
    bleuscore_4_gram.append(nltk.translate.bleu_score.sentence_bleu([target_reference[i]], hypothesis[i],weights=(0.25,0.25,0.25,0.25)))
f2=open("Predictions.txt","w+")
for j in range(10000):

    f2.write(' Source :'+' '.join([i for i in source_reference[j] if i not in ['</s>']]))
    f2.write(' Truth : '+' '.join([index2target[i] for i in target_reference[j].data.tolist()[0] if i not in [2, 3]]))
    f2.write(' Prediction : '+' '.join([i for i in hypothesis[j] if i not in ['</s>']]))

    f2.write("\n")


bleuscore_unigram_t=[]
bleuscore_4_gram_t=[]
for i in range(10000):
    bleuscore_unigram_t.append(nltk.translate.bleu_score.sentence_bleu([target_reference_t[i]], hypothesis_t[i],weights=(1,0,0,0)))
    bleuscore_4_gram_t.append(nltk.translate.bleu_score.sentence_bleu([target_reference_t[i]], hypothesis_t[i],weights=(0.25,0.25,0.25,0.25)))
f3=open("Predictions_t.txt","w+")
for j in range(10000):

    f3.write(' Source :'+' '.join([i for i in source_reference_t[j] if i not in ['</s>']]))
    f3.write(' Truth : '+' '.join([index2target[i] for i in target_reference_t[j].data.tolist()[0] if i not in [2, 3]]))
    f3.write(' Prediction : '+' '.join([i for i in hypothesis_t[j] if i not in ['</s>']]))

    f3.write("\n")

f2.close()

print(bleuscore_unigram)
f= open("bleuscore.txt","w+")
f.write("Unigram Bleu Score "+str(float(np.mean(bleuscore_unigram_t))))
f.write("4 gram Bleu Score "+str(float(np.mean(bleuscore_4_gram_t))))

if USE_CUDA:
    attn = attn.cpu()
print("Saving attention diagram")
show_attention(source_reference[0], hypothesis[0], attn.data)
