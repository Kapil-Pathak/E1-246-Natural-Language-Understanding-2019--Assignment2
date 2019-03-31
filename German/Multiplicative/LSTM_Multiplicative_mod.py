from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


from google.colab import drive
drive.mount('/content/drive')
lines=open('/content/drive/My Drive/data/English.txt', 'r').readlines()
print(len(lines))

f1=open("/content/drive/My Drive/data/German.txt", "r")
lines1=f1.readlines()
print(len(lines1))

USE_CUDA = torch.cuda.is_available()
gpus = [0]
torch.cuda.set_device(gpus[0])

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

eng=[]
from tqdm import tqdm_notebook as tqdm
for line in tqdm(lines[1:]):
    s = re.sub(r"([.!?])", r"", line.lower())
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    eng.append(s)

training_words_eng=[]
for line in tqdm(eng):
  for words in line.split():
    training_words_eng.append(words)

from collections import Counter
count_eng=dict()
for word in tqdm(training_words_eng):
  count_eng[word]=count_eng.get(word,0)+1

sent_preprocessed=[]
length=0
removed=0
for sent in tqdm(eng):
    endsent=sent.split()
    for word in endsent:
      if count_eng[word]<=5:
        endsent.remove(word)
        removed+=1
    endsent.append('EOS')
    length+=len(endsent)
    sent_preprocessed.append(endsent)
avg_length=length/len(eng)

del training_words_eng, lines, eng, count_eng

german=[]
from tqdm import tqdm_notebook as tqdm
for line in tqdm(lines1[1:]):
    s = re.sub(r"([.!?])", r" \1", line.lower())
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    german.append(s)

training_words_ger=[]
for line in tqdm(german):
  for words in line.split():
    training_words_ger.append(words)

count_ger=dict()
for word in tqdm(training_words_ger):
  count_ger[word]=count_ger.get(word,0)+1

del training_words_ger, lines1

sent_preprocessed_ger=[]
for sent in tqdm(german):
    endsent=sent.split()
    for word in endsent:
      if count_ger[word]<=5:
        endsent.remove(word)
        removed+=1
    endsent.append('EOS')
    sent_preprocessed_ger.append(endsent)

vocab_eng=set()
for sent in tqdm(sent_preprocessed):
    for words in sent:
        vocab_eng.add(words)

index = 2
engwords2ind=dict()
engwords2ind['EOS']=0
engwords2ind['/PAD']=1
for word in vocab_eng:
  if word is not 'EOS':
    engwords2ind[word] = index
    index += 1
ind2engwords = dict(zip(engwords2ind.values(), engwords2ind.keys()))

s=[]
for sent in tqdm(sent_preprocessed):
  s_sent=[]
  for word in sent:
    s_sent.append(engwords2ind[word])
  s.append(Variable(LongTensor(s_sent)))

vocab_ger=set()
for sent in tqdm(sent_preprocessed_ger):
    for words in sent:
        vocab_ger.add(words)

index = 2
gerwords2ind=dict()
gerwords2ind['EOS']=0
gerwords2ind['/PAD']=1
gerwords2ind['SOS']=2
for word in vocab_ger:
  if word is not 'EOS':
    gerwords2ind[word] = index
    index += 1
ind2gerwords = dict(zip(gerwords2ind.values(), gerwords2ind.keys()))

t=[]
for sent in tqdm(sent_preprocessed_ger):
  t_sent=[]
  for word in sent:
    t_sent.append(gerwords2ind[word])
  t.append(Variable(LongTensor(t_sent)))

del sent_preprocessed_ger

def filter_by_maxlen(s_data,t_data,maxlen,engwords2ind,gerwords2ind):
  f_s=[]
  f_t=[]
  input_length=[]
  target_length=[]
  for i in range(len(t_data)):

    if len(s_data[i])<maxlen and len(t_data[i])<maxlen and len(s_data[i])>2 and len(t_data[i])>2:
      input_length.append(len(s_data[i]))
      target_length.append(len(t_data[i]))
      s_data[i]=torch.cat([s_data[i].view(1,-1), Variable(LongTensor([engwords2ind['/PAD']] * (maxlen - len(s_data[i])))).view(1, -1)], 1)
      t_data[i]=torch.cat([t_data[i].view(1,-1), Variable(LongTensor([gerwords2ind['/PAD']] * (maxlen - len(t_data[i])))).view(1, -1)], 1)
      f_s.append(s_data[i])
      f_t.append(t_data[i])
  return f_s, f_t, input_length, target_length

fil_s, fil_t,l,m=filter_by_maxlen(s,t,30,engwords2ind,gerwords2ind)

def getBatch(batch_size, source_data, target_data,l,m):

    sindex=0
    while (sindex+batch_size)< len(source_data):
        batch_source = source_data[sindex: sindex+batch_size]
        batch_target= target_data[sindex: sindex+batch_size]
        ls=l[sindex: sindex+batch_size]
        lm=m[sindex: sindex+batch_size]
        sindex=sindex+batch_size
        ls[:] = [x - 1 for x in ls]
        lm[:] = [x - 1 for x in lm]
        ls_reverse=[-x for x in ls]
        lm_reverse=[-x for x in lm]

        so=np.argsort(ls_reverse)
        to=np.argsort(lm_reverse)
        ls=[ls[i] for i in so]
        lm=[lm[i] for i in to]
        batch_source = [batch_source[i] for i in so]
        batch_target = [batch_target[i] for i in to]
        yield batch_source, batch_target,ls,lm

    if sindex+batch_size >= len(source_data):
        batch_source = source_data[sindex:]
        batch_target = target_data[sindex:]
        ls=l[sindex:]
        lm=m[sindex:]
        ls[:] = [x - 1 for x in ls]
        lm[:] = [x - 1 for x in lm]
        ls_reverse=[-x for x in ls]
        lm_reverse=[-x for x in lm]

        so=np.argsort(ls_reverse)
        to=np.argsort(lm_reverse)
        ls=[ls[i] for i in so]
        lm=[lm[i] for i in to]
        batch_source = [batch_source[i] for i in so]
        batch_target = [batch_target[i] for i in to]
        yield batch_source, batch_target,ls,lm


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size,hidden_size, n_layers=1):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.n_direction = 2
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, batch_first=True, bidirectional=True)



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
        self.attn_1 = nn.Linear(self.hidden_size, self.hidden_size) # Attention
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
        energies = self.attn_1(encoder_outputs.contiguous().view(batch_size * max_len, -1)) # B*T,D -> B*T,D
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


    def Key_ValueAttention(self, hidden, encoder_outputs, encoder_maskings):
        """
        hidden : 1,B,D
        encoder_outputs : B,T,D
        encoder_maskings : B,T # ByteTensor
        """
        #hidden = hidden[0].unsqueeze(2)  # (1,B,D) -> (B,D,1)

        batch_size = encoder_outputs.size(0) # B
        max_len= encoder_outputs.size(1) # T
        energies_1 = self.attn_1(encoder_outputs.contiguous().view(batch_size * max_len, -1)) # B*T,D -> B*T,D
        energies_1=self.tanh(energies_1)
        energies_2= self.attn_2(hidden)
        energies_2=self.tanh(energies_2)
        energies_1 = energies_1.view(batch_size,max_len, -1) # B,T,D
        energies=energies_2[0].unsqueeze(2)
        #energies = self.tanh(energies)
        #energies = self.attn_v(energies)
        attn_energies = energies_1.bmm(energies).squeeze(2) # B,T,D * B,D,1 --> B,T

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
            context, alpha = self.Multiplicative_Attention(hidden, encoder_outputs, encoder_maskings)

        #  column-wise concat, reshape!!
        scores = torch.cat(decode, 1)
        return scores.view(inputs.size(0) * max_length, -1)

    def decode(self, context, encoder_outputs):
        start_decode = Variable(LongTensor([[target2index['SOS']] * 1])).transpose(0, 1)
        embedded = self.embedding(start_decode)
        hidden = self.init_hidden(start_decode)
        context_c = self.initContext(start_decode)
        decodes = []
        attentions = []
        decoded = embedded
        max_len=25
        i=0
        while decoded.data.tolist()[0] != target2index['EOS']and i<max_len: # until </s>
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
            context, alpha = self.Multiplicative_Attention(hidden, encoder_outputs,None)
            attentions.append(alpha.squeeze(1))

        return torch.cat(decodes).max(1)[1], torch.cat(attentions)

EPOCH =20
BATCH_SIZE = 256
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 512
LR = 0.001
DECODER_LEARNING_RATIO = 5.0
RESCHEDULED = False

encoder = Encoder(len(source2index), EMBEDDING_SIZE, HIDDEN_SIZE, 3)
decoder = Decoder(len(target2index), EMBEDDING_SIZE, HIDDEN_SIZE * 2)
encoder.init_weight()
decoder.init_weight()

encoder = encoder.cuda()
decoder = decoder.cuda()

loss_function = nn.CrossEntropyLoss(ignore_index=0)
enc_optimizer = optim.Adam(encoder.parameters(), lr=LR)
dec_optimizer = optim.Adam(decoder.parameters(), lr=LR * DECODER_LEARNING_RATIO)

def process_batch(batch_s, batch_t):
  batch_s =  sorted(batch_s, key=lambda b:b.size(1), reverse=True)
  for j in range(len(batch_s)):
    if j==0:
      batch_s2=batch_s[j]
      batch_t2=batch_t[j]
    else:
      batch_s[j]=batch_s[j].view(1,-1)
      batch_t[j]=batch_t[j].view(1,-1)
      batch_s2=torch.cat((batch_s2,batch_s[j]))
      batch_t2=torch.cat((batch_t2,batch_t[j]))
  return batch_s2,batch_t2



f1=open("Loss.txt","w+")
from tqdm import tqdm_notebook as tqdm
torch.backends.cudnn.benchmark=True
ffrom torch.nn.utils.rnn import PackedSequence,pack_padded_sequence


for epoch in tqdm(range(EPOCH)):
    losses=[]
    for i, (batch_s, batch_t,input_lengths,target_lengths) in enumerate(getBatch(BATCH_SIZE,fil_s,fil_t,l,m)):
        #inputs,targets,input_lengths,target_lengths=process_batch(batch_s, batch_t)

        inputs,targets=process_batch(batch_s, batch_t)

        input_masks = torch.cat([Variable(ByteTensor(tuple(map(lambda s: s ==1, t.data)))) for t in inputs]).view(inputs.size(0), -1)
        start_decode = Variable(LongTensor([[gerwords2ind['SOS']] * targets.size(0)])).transpose(0, 1)
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


MODEL_SAVE_PATH="/home/kapil/Key_Value/encoder.pt"
#torch.save(decoder.state_dict(), MODEL_SAVE_PATH)
torch.save(encoder.state_dict(), MODEL_SAVE_PATH)
MODEL_SAVE_PATH1="/home/kapil/Key_Value/decoder.pt"
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
    plt.savefig('LSTM_Multiplicative.png')
    plt.close()

target_reference=[]
hypothesis=[]
source_reference=[]

print("Initializing testing......")
for i in range(100):
    print(i)
    #test = random.choice(test_data)
    #test = random.choice(train_data)
    input_ = fil_s[i]
    truth = fil_t[i]
    #print(input_)
    #print(len(input_))
    output, hidden, context_c = encoder(input_, [input_.size(1)])
    pred, attn = decoder.decode(hidden, output)

    input_ = [index2source[i] for i in input_.data.tolist()[0]]
    pred = [index2target[i] for i in pred.data.tolist()]
    source_reference.append(input_)
    target_reference.append(truth)
    hypothesis.append(pred)

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
"""
f2.write(' Source :'+' '.join([i for i in source_reference[1] if i not in ['</s>']]))
f2.write(' Truth : '+' '.join([index2target[i] for i in target_reference[1].data.tolist()[0] if i not in [2, 3]]))
f2.write(' Prediction : '+' '.join([i for i in hypothesis[1] if i not in ['</s>']]))

f2.write("\n")

f2.write(' Source :'+' '.join([i for i in source_reference[2] if i not in ['</s>']]))
f2.write(' Truth : '+' '.join([index2target[i] for i in target_reference[2].data.tolist()[0] if i not in [2, 3]]))
f2.write(' Prediction : '+' '.join([i for i in hypothesis[2] if i not in ['</s>']]))

f2.write("\n")

f2.write(' Source :'+' '.join([i for i in source_reference[3] if i not in ['</s>']]))
f2.write(' Truth : '+' '.join([index2target[i] for i in target_reference[3].data.tolist()[0] if i not in [2, 3]]))
f2.write(' Prediction : '+' '.join([i for i in hypothesis[3] if i not in ['</s>']]))

f2.write("\n")

f2.write(' Source :'+' '.join([i for i in source_reference[4] if i not in ['</s>']]))
f2.write(' Truth : '+' '.join([index2target[i] for i in target_reference[4].data.tolist()[0] if i not in [2, 3]]))
f2.write(' Prediction : '+' '.join([i for i in hypothesis[4] if i not in ['</s>']]))

f2.write("\n")

f2.write(' Source :'+' '.join([i for i in source_reference[5] if i not in ['</s>']]))
f2.write(' Truth : '+' '.join([index2target[i] for i in target_reference[5].data.tolist()[0] if i not in [2, 3]]))
f2.write(' Prediction : '+' '.join([i for i in hypothesis[5] if i not in ['</s>']]))

f2.write("\n")

f2.write(' Source :'+' '.join([i for i in source_reference[6] if i not in ['</s>']]))
f2.write(' Truth : '+' '.join([index2target[i] for i in target_reference[6].data.tolist()[0] if i not in [2, 3]]))
f2.write(' Prediction : '+' '.join([i for i in hypothesis[6] if i not in ['</s>']]))

f2.write("\n")

f2.write(' Source :'+' '.join([i for i in source_reference[7] if i not in ['</s>']]))
f2.write(' Truth : '+' '.join([index2target[i] for i in target_reference[7].data.tolist()[0] if i not in [2, 3]]))
f2.write(' Prediction : '+' '.join([i for i in hypothesis[7] if i not in ['</s>']]))

f2.write("\n")

f2.write(' Source :'+' '.join([i for i in source_reference[8] if i not in ['</s>']]))
f2.write(' Truth : '+' '.join([index2target[i] for i in target_reference[8].data.tolist()[0] if i not in [2, 3]]))
f2.write(' Prediction : '+' '.join([i for i in hypothesis[8] if i not in ['</s>']]))

f2.write("\n")

f2.write(' Source :'+' '.join([i for i in source_reference[9] if i not in ['</s>']]))
f2.write(' Truth : '+' '.join([index2target[i] for i in target_reference[9].data.tolist()[0] if i not in [2, 3]]))
f2.write(' Prediction : '+' '.join([i for i in hypothesis[9] if i not in ['</s>']]))
"""
f2.close()
print('Source :',' '.join([i for i in source_reference[0] if i not in ['</s>']]))
print('Truth : ',' '.join([index2target[i] for i in target_reference[0].data.tolist()[0] if i not in [2, 3]]))
print('Prediction : ',' '.join([i for i in hypothesis[0] if i not in ['</s>']]))

print('Source :',' '.join([i for i in source_reference[1] if i not in ['</s>']]))
print('Truth : ',' '.join([index2target[i] for i in target_reference[1].data.tolist()[0] if i not in [2, 3]]))
print('Prediction : ',' '.join([i for i in hypothesis[1] if i not in ['</s>']]))

print('Source :',' '.join([i for i in source_reference[2] if i not in ['</s>']]))
print('Truth : ',' '.join([index2target[i] for i in target_reference[0].data.tolist()[0] if i not in [2, 3]]))
print('Prediction : ',' '.join([i for i in hypothesis[2] if i not in ['</s>']]))

print('Source :',' '.join([i for i in source_reference[3] if i not in ['</s>']]))
print('Truth : ',' '.join([index2target[i] for i in target_reference[3].data.tolist()[0] if i not in [2, 3]]))
print('Prediction : ',' '.join([i for i in hypothesis[3] if i not in ['</s>']]))

print('Source :',' '.join([i for i in source_reference[4] if i not in ['</s>']]))
print('Truth : ',' '.join([index2target[i] for i in target_reference[4].data.tolist()[0] if i not in [2, 3]]))
print('Prediction : ',' '.join([i for i in hypothesis[4] if i not in ['</s>']]))

print('Source :',' '.join([i for i in source_reference[5] if i not in ['</s>']]))
print('Truth : ',' '.join([index2target[i] for i in target_reference[5].data.tolist()[0] if i not in [2, 3]]))
print('Prediction : ',' '.join([i for i in hypothesis[5] if i not in ['</s>']]))

print('Source :',' '.join([i for i in source_reference[6] if i not in ['</s>']]))
print('Truth : ',' '.join([index2target[i] for i in target_reference[6].data.tolist()[0] if i not in [2, 3]]))
print('Prediction : ',' '.join([i for i in hypothesis[6] if i not in ['</s>']]))

print('Source :',' '.join([i for i in source_reference[7] if i not in ['</s>']]))
print('Truth : ',' '.join([index2target[i] for i in target_reference[7].data.tolist()[0] if i not in [2, 3]]))
print('Prediction : ',' '.join([i for i in hypothesis[7] if i not in ['</s>']]))

print('Source :',' '.join([i for i in source_reference[8] if i not in ['</s>']]))
print('Truth : ',' '.join([index2target[i] for i in target_reference[8].data.tolist()[0] if i not in [2, 3]]))
print('Prediction : ',' '.join([i for i in hypothesis[8] if i not in ['</s>']]))

print('Source :',' '.join([i for i in source_reference[9] if i not in ['</s>']]))
print('Truth : ',' '.join([index2target[i] for i in target_reference[9].data.tolist()[0] if i not in [2, 3]]))
print('Prediction : ',' '.join([i for i in hypothesis[9] if i not in ['</s>']]))
print(bleuscore_unigram)
f= open("bleuscore.txt","w+")
f.write("Unigram Bleu Score "+str(float(np.mean(bleuscore_unigram))))
f.write("4 gram Bleu Score "+str(float(np.mean(bleuscore_4_gram))))

if USE_CUDA:
    attn = attn.cpu()
print("Saving attention diagram")
show_attention(source_reference[0], hypothesis[0], attn.data)
