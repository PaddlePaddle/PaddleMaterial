#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, Concatenate
from keras import regularizers
from keras.callbacks import History, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

smifile = "Modeldata.csv"
data = pd.read_csv(smifile)
smiles = data["SMILES"].values
target_properties = data[["ATSC1pe", "MATS2c", "SlogP_VSA2"]].values

smiles_train, smiles_test, y_train, y_test = train_test_split(smiles, target_properties, random_state=42)


# In[2]:


volcabulary = set("".join(list(data.SMILES)))
volcabulary.update({'<go>', '<eos>'})
docs = dict((c,i) for i,c in enumerate(volcabulary))
docs_swap= {v: k for k, v in docs.items()}
embed = max([len(smile) for smile in data.SMILES]) + 5


# In[3]:


def vectorize(smiles):
    one_hot = np.zeros((smiles.shape[0], embed, len(volcabulary)), dtype=np.int8)
    for i, smile in enumerate(smiles):
        one_hot[i, 0, docs['<go>']] = 1
        for j, c in enumerate(smile):
            one_hot[i, j + 1, docs[c]] = 1
        one_hot[i, len(smile) + 1:, docs['<eos>']] = 1
    return one_hot[:, :-1, :], one_hot[:, 1:, :]

X_train, Y_train = vectorize(smiles_train)
X_test, Y_test = vectorize(smiles_test)

print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")
print(f"y_train shape: {y_train.shape}")


# In[4]:


"".join([docs_swap[idx] for idx in np.argmax(X_train[0,:,:], axis=1)])


# In[5]:


"".join([docs_swap[idx] for idx in np.argmax(Y_train[0,:,:], axis=1)])


# ## LSTM

# In[6]:


input_shape = X_train.shape[1:]
output_dim = Y_train.shape[-1]
latent_dim = 64
lstm_dim = 128
unroll = False

encoder_inputs = Input(shape=input_shape)
encoder = LSTM(lstm_dim, return_state=True, unroll=unroll)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

property_input = Input(shape=(3,), name="property_input")  # 4个目标性质
merged = Concatenate(axis=-1)([state_h, state_c, property_input])  # 合并状态和性质

neck = Dense(latent_dim, activation="relu", kernel_regularizer=regularizers.l2(0.001))
neck_outputs = neck(merged)

decode_h = Dense(lstm_dim, activation="relu")
decode_c = Dense(lstm_dim, activation="relu")
state_h_decoded = decode_h(neck_outputs)
state_c_decoded = decode_c(neck_outputs)
encoder_states = [state_h_decoded, state_c_decoded]

decoder_inputs = Input(shape=input_shape)
decoder_lstm = LSTM(lstm_dim, return_sequences=True, unroll=unroll)
decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(output_dim, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, property_input, decoder_inputs], decoder_outputs)
model.summary()


# # 模型训练

# In[7]:


h = History()
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=5, min_lr=0.000001, verbose=1, min_delta=1e-6)
opt = Adam(learning_rate=0.001) #Default 0.001
model.compile(optimizer=opt, loss='categorical_crossentropy')
early_stopping = EarlyStopping(
    monitor="val_loss",  
    patience=5,          
    restore_best_weights=True,  
    verbose=1            
)

model.fit([X_train, y_train, X_train], Y_train,
          epochs=60,
          batch_size=512,
          shuffle=True,
          callbacks=[h, rlr, early_stopping],
          validation_data=[[X_test, y_test, X_test], Y_test])


plt.plot(h.history['loss'], label='Training loss')
plt.plot(h.history['val_loss'], label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()


plt.show()


# ## 保存模型

# In[79]:


model.save('Seq2Seq_model.keras') 


# ## 加载模型

# In[14]:


import tensorflow as tf
loaded_model = tf.keras.models.load_model('Seq2Seq_model.keras')


# ## 潜在空间生成分子

# In[8]:


smiles_to_latent_model = Model([encoder_inputs, property_input], neck_outputs)


latent_input = Input(shape=(latent_dim,))
state_h_decoded_2 = decode_h(latent_input)
state_c_decoded_2 = decode_c(latent_input)
latent_to_states_model = Model(latent_input, [state_h_decoded_2, state_c_decoded_2])

inf_decoder_inputs = Input(batch_shape=(1, 1, input_shape[1]))
inf_decoder_lstm = LSTM(lstm_dim,
                        return_sequences=True,
                        unroll=unroll,
                        stateful=True)
inf_decoder_outputs = inf_decoder_lstm(inf_decoder_inputs)
inf_decoder_dense = Dense(output_dim, activation='softmax')
inf_decoder_outputs = inf_decoder_dense(inf_decoder_outputs)
sample_model = Model(inf_decoder_inputs, inf_decoder_outputs)


# In[9]:


for i in range(1, 3):  # i 是解码器相关的LSTM层（具体的层数需要根据你模型的具体结构来调整）
    sample_model.layers[i].set_weights(model.layers[i+7].get_weights())
sample_model.summary()


# In[10]:


x_latent = smiles_to_latent_model.predict([X_test, y_test])


# In[74]:


def latent_to_smiles(latent, property_value):
    states = latent_to_states_model.predict(latent, verbose=0)
    startidx = docs['<go>']
    samplevec = np.zeros((1, 1, len(docs)))  # 不要硬编码 22，使用 len(docs)
    samplevec[0, 0, startidx] = 1
    smiles = ""

    state_h, state_c = states

    for i in range(28):
        o = sample_model.predict(samplevec, verbose=0)
        sampleidx = np.argmax(o)
        samplechar = docs_swap[sampleidx]
        if samplechar != '<eos>':
            smiles = smiles + samplechar
            samplevec = np.zeros((1, 1, len(docs)))
            samplevec[0, 0, sampleidx] = 1
        else:
            break
    return smiles


# In[75]:


property_value = np.array([0, -0.2, 20])


# In[93]:


latent1 = x_latent[2:3]
latent0 = x_latent[8:9]
mols1 = []
ratios = np.linspace(0,1,30)
for r in ratios:
    rlatent = (1.0 - r) * latent0 + r * latent1
    smiles = latent_to_smiles(rlatent, property_value)
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mols1.append(mol)
    else:
        print(smiles)

Draw.MolsToGridImage(mols1, molsPerRow=5)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




