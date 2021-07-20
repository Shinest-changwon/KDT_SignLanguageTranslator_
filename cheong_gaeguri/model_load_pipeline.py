#!/usr/bin/env python
# coding: utf-8

# ## 데이터 로드 및 전처리

# In[1]:


import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython import get_ipython

#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
 
import matplotlib.font_manager as fm
fontpath = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
font = fm.FontProperties(fname=fontpath, size=9)
plt.rc('font', family='NanumBarunGothic') 
mpl.font_manager._rebuild()


# In[2]:


import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

import time
import re
import os
import io
import pandas as pd
# print(tf.__version__)


# In[4]:


# path = 'C:\python\KDT\sen_label.xlsx'# 윈도우 경로
path = os.path.realpath(__file__)[:-38]+"/label_csv/sen_label.xlsx"
# path = "../label_csv/sen_label.xlsx"
data = pd.read_excel(path, engine='openpyxl')
data.head(10)


# In[5]:


import math
morp_sen = []
for i in range(len(data)):
    sentence = []
    for no in range(1, 8):
        word = 'morp{}'.format(no)
        
        
        if pd.isna(data[word][i]) != True:
#         if 'NaN' not in word or math.isnan(float('{}'.format(word))) != True:

            sentence.append(str(data[word][i]))
            # print(sentence)
    a = ' '.join(sentence)
    morp_sen.append(a)

# In[6]:


kor_sen = data['한국어 번역']


# In[7]:


cleaned_corpus_eng = morp_sen
cleaned_corpus_kor = kor_sen


# In[8]:


def preprocess_sentence(sentence, s_token=False, e_token=False):
    sentence = sentence.lower().strip()

    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)# \1: 그루핑된 문자열 재참조하기(문장기호가 있으면 앞뒤로 공백 후 재참조
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"[^a-zA-Z가-힣0-9?.!,]+", " ", sentence)#알파벳 대소문자, 한글, 숫자, 필수 기호만 포함

    sentence = sentence.strip()

    if s_token:
        sentence = '<start> ' + sentence

    if e_token:
        sentence += ' <end>'
    
    return sentence


# In[9]:


enc_corpus = []
dec_corpus = []

for eng, kor in zip(cleaned_corpus_eng, cleaned_corpus_kor): #[:num_examples]:
#     eng, spa = pair.split("\t")

    enc_corpus.append(preprocess_sentence(kor))
    dec_corpus.append(preprocess_sentence(eng, s_token=True, e_token=True))

# print("Korean:", enc_corpus[100])   # 1번 승강장입니다 .
# print("English:", dec_corpus[100])   # <start> 지하철 곳 번호 1 <end>


# In[10]:


# print(len(enc_corpus))


# In[11]:


from konlpy.tag import Mecab
mecab = Mecab()
tokenized_kor = []
tokenized_eng = []

for sen_kor, sen_eng in zip(enc_corpus, dec_corpus):
    proto_kor = (mecab.morphs(sen_kor))
    proto_eng = sen_eng.split()
    if len(proto_kor) <= 20 and len(proto_eng) <= 20:
#         proto_kor.insert(0, "<start>")
#         proto_kor.append("<end>")
        tokenized_kor.append(proto_kor)
        tokenized_eng.append(proto_eng)


# In[12]:


# for i, j in zip(tokenized_kor[:5], tokenized_eng[:5]):
    # print(i, j)


# In[13]:


def tokenize(corpus):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer.fit_on_texts(corpus)

    tensor = tokenizer.texts_to_sequences(corpus)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    return tensor, tokenizer


# In[14]:


enc_tensor, enc_tokenizer = tokenize(tokenized_kor)
dec_tensor, dec_tokenizer = tokenize(tokenized_eng)

enc_train, dec_train = enc_tensor, dec_tensor

# print("Korean Vocab Size:", len(enc_tokenizer.index_word))
# print("English Vocab Size:", len(dec_tokenizer.index_word))


# In[15]:


# print(enc_tensor.shape)
# print(enc_tensor[0])


# In[16]:


# print(dec_tensor.shape)
# print(dec_tensor[0])


# In[17]:


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        
        # keras.layers.dense는 인자로 받은 벡터와 자체적으로 만든 가중치를 dot해서 결과로 return함
        # output = activation(dot(input, kernel) + bias)
        
        super(BahdanauAttention, self).__init__()
        self.w_enc = tf.keras.layers.Dense(units)# 인코더 레이어 정의
        self.w_dec = tf.keras.layers.Dense(units)# 디코더 레이어 정의
        self.w_com = tf.keras.layers.Dense(1)
    
    def call(self, h_enc, h_dec):
        # h_enc shape: [batch x length x units]
        # h_dec shape: [batch x units]

        h_enc = self.w_enc(h_enc)# h_enc를 받아서 연산 후 h값 출력
        h_dec = tf.expand_dims(h_dec, 1)# h_dec을 받아서 형태를 맞추기 위해 expand_dims 사용, 인자로 인덱스를 받으며 1은 두번째 차원을 확장
        h_dec = self.w_dec(h_dec)

        score = self.w_com(tf.nn.tanh(h_dec + h_enc))
        
        attn = tf.nn.softmax(score, axis=1)

        context_vec = attn * h_enc
        context_vec = tf.reduce_sum(context_vec, axis=1)

        return context_vec, attn


# In[19]:


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()

        self.enc_units = enc_units
        
        # keras.layers.Embedding(input_dim, output_dim)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(enc_units,
                                       return_sequences=True)# 시퀀스의 마지막 값을 반환할 것인가

    def call(self, x):
        out = self.embedding(x)
        out = self.gru(out)

        return out


# In[20]:


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(dec_units,
                                       return_sequences=True,
                                       return_state=True)
        self.fc = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, h_dec, enc_out):
        context_vec, attn = self.attention(enc_out, h_dec)

        out = self.embedding(x)
        out = tf.concat([tf.expand_dims(context_vec, 1), out], axis=-1)
#         print(out.shape)
        out, h_dec = self.gru(out)
        out = tf.reshape(out, (-1, out.shape[2]))
        out = self.fc(out)

        return out, h_dec, attn


# In[21]:


# 코드를 실행하세요.

BATCH_SIZE     = 8
SRC_VOCAB_SIZE = len(enc_tokenizer.index_word) + 1 # 예: len(enc_tokenizer.index_word) + 1
TGT_VOCAB_SIZE = len(dec_tokenizer.index_word) + 1 # 예: len(dec_tokenizer.index_word) + 1

units         = 128
embedding_dim = 256

# print(SRC_VOCAB_SIZE, embedding_dim, units)
encoder = Encoder(SRC_VOCAB_SIZE, embedding_dim, units)
print("hi")
decoder = Decoder(TGT_VOCAB_SIZE, embedding_dim, units)

# sample input
sequence_len = 15

sample_enc = tf.random.uniform((BATCH_SIZE, sequence_len))
sample_output = encoder(sample_enc)

# print ('Encoder Output:', sample_output.shape)

sample_state = tf.random.uniform((BATCH_SIZE, units))

sample_logits, h_dec, attn = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                     sample_state, sample_output)


# print ('Decoder Output:', sample_logits.shape)
# print ('Decoder Hidden State:', h_dec.shape)
# print ('Attention:', attn.shape)


# In[22]:


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss = loss_object(real, pred)
    
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    
    return tf.reduce_mean(loss)


# In[23]:


import os
import tensorflow as tf
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


# In[24]:


eval_checkpoint_dir = './eval_training_checkpoints'
eval_checkpoint_prefix = os.path.join(eval_checkpoint_dir, "ckpt")
eval_checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


# In[25]:


# @tf.function 데코레이터는 훈련 외적인 텐서플로우 연산을 GPU에서 동작하게 해 훈련을 가속
@tf.function
def train_step(src, tgt, encoder, decoder, optimizer, dec_tok):
    bsz = src.shape[0]
    loss = 0

    
    # tf.GradientTape()는 학습하며 발생한 모든 연산을 기록하는 테이프
    with tf.GradientTape() as tape:
        enc_out = encoder(src)
        h_dec = enc_out[:, -1]# enc_hidden
        
        dec_src = tf.expand_dims([dec_tok.word_index['<start>']] * bsz, 1)

        for t in range(1, tgt.shape[1]):
            pred, h_dec, _ = decoder(dec_src, h_dec, enc_out)

            loss += loss_function(tgt[:, t], pred)
            # 교사강요(teacher forcing)
            dec_src = tf.expand_dims(tgt[:, t], 1)
        
    batch_loss = (loss / int(tgt.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    
    return batch_loss


# In[26]:


from tensorflow.python.client import device_lib
device_lib.list_local_devices()


# In[27]:


# eval_step() 정의하기
# Define eval_step
from tqdm import tqdm    # tqdm
import random


@tf.function
def eval_step(src, tgt, encoder, decoder, dec_tok):
    bsz = src.shape[0]
    loss = 0

    enc_out = encoder(src)

    h_dec = enc_out[:, -1]

    dec_src = tf.expand_dims([dec_tok.word_index['<start>']] * bsz, 1)

    for t in range(1, tgt.shape[1]):
        pred, h_dec, _ = decoder(dec_src, h_dec, enc_out)

        loss += loss_function(tgt[:, t], pred)
        dec_src = tf.expand_dims(tgt[:, t], 1)

    batch_loss = (loss / int(tgt.shape[1]))

    return batch_loss


# In[28]:


def evaluate(sentence, encoder, decoder):
    attention = np.zeros((dec_train.shape[-1], enc_train.shape[-1]))
    
    sentence = preprocess_sentence(sentence)
    inputs = enc_tokenizer.texts_to_sequences([mecab.morphs(sentence)])
    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
                                                           maxlen=enc_train.shape[-1],
                                                           padding='post')

    result = ''

    enc_out = encoder(inputs)

    dec_hidden = enc_out[:, -1]
    dec_input = tf.expand_dims([dec_tokenizer.word_index['<start>']], 0)

    for t in range(dec_train.shape[-1]):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention[t] = attention_weights.numpy()

        predicted_id =         tf.argmax(tf.math.softmax(predictions, axis=-1)[0]).numpy()

        result += dec_tokenizer.index_word[predicted_id] + ' '

        if dec_tokenizer.index_word[predicted_id] == '<end>':
            return result, sentence, attention

        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention



def translate(sentence, encoder, decoder):
    result, sentence, attention = evaluate(sentence, encoder, decoder)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))
    
    attention = attention[:len(result.split()), :len(sentence.split())]
    return result

# In[29]:


# checkpoint_dir내에 있는 최근 체크포인트(checkpoint)를 복원합니다.
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


# In[30]:


# checkpoint_dir내에 있는 최근 체크포인트(checkpoint)를 복원합니다.
checkpoint.restore(tf.train.latest_checkpoint(eval_checkpoint_dir))


# In[31]:


# translate("손전등을 잃어버렸습니다.", encoder, decoder)


# In[32]:


# translate("안녕하세요.", encoder, decoder)
# translate("1호선을 타는 곳은 어디인가요?", encoder, decoder)
# translate("여기서 1호선을 탈 수 있습니다.", encoder, decoder)

# translate("(서울대학교)방향으로 가려면 어떻게 가나요?", encoder, decoder)
# translate("지하철 갈아타는 곳으로 안내해드릴까요?", encoder, decoder)
# translate("감사합니다.", encoder, decoder)

# translate("가방을 잃어버렸습니다.", encoder, decoder)
# translate("어디요?", encoder, decoder)
from konlpy.tag import Mecab

def pipeline(sent):
    # mecab = Mecab()



    BATCH_SIZE     = 8
    units         = 128
    embedding_dim = 256
    SRC_VOCAB_SIZE = 483
    TGT_VOCAB_SIZE = 321


    # print(SRC_VOCAB_SIZE, embedding_dim, units)
    encoder = Encoder(SRC_VOCAB_SIZE, embedding_dim, units)
    decoder = Decoder(TGT_VOCAB_SIZE, embedding_dim, units)
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    checkpoint_dir = '/home/aiffel-dj16/dev/KDT_SignLanguageTranslator/cheong_gaeguri/training_checkpoints'
    # checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                    encoder=encoder,
                                    decoder=decoder)
    print("-----------------------------"+checkpoint_prefix)
    
    # eval_checkpoint_dir = './eval_training_checkpoints'
    eval_checkpoint_dir = '/home/aiffel-dj16/dev/KDT_SignLanguageTranslator/cheong_gaeguri/eval_training_checkpoints'
    eval_checkpoint_prefix = os.path.join(eval_checkpoint_dir, "ckpt")
    print("-----------------------"+eval_checkpoint_prefix)
    eval_checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                    encoder=encoder,
                                    decoder=decoder)
    # checkpoint_dir내에 있는 최근 체크포인트(checkpoint)를 복원합니다.
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


    # In[30]:


    # checkpoint_dir내에 있는 최근 체크포인트(checkpoint)를 복원합니다.
    checkpoint.restore(tf.train.latest_checkpoint(eval_checkpoint_dir))
    result = translate(sent, encoder, decoder)

    return result
    
# %%
