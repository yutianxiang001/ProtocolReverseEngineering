import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import activations
from tensorflow.keras.layers import Layer, Input, Embedding, LSTM, Dense, Attention, Bidirectional, SimpleRNN, GRU
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers.recurrent import RNN

pcap_file_path='F:\\dataset\\http_thy.pcap'

app_text=[]
app_ascii_bin=[]
low_dimension_feature=[]

tem=[]
""" with open(pcap_file_path, mode='rb') as f:
    for line in f:
        print(line) """

    #目前将数据部分暂缓，拿到的数据为切割好的应用层文本数据列表，编码为ascii
for i in app_text:
    for j in i:
        tem.append(ord(j))
    app_ascii_bin.append(tem)
    tem=[]
class encoder(keras.Model):
    def __init__(self):
        super(encoder,self).__init__()

        self.layer1=Dense(128,activation='relu',name='encode_layer1')
        self.layer2=Dense(64,activation='relu',name='encode_layer2')
        self.layer3=Dense(32,activation='relu',name='encode_layer3')
        self.layer4=Dense(2,activation='sigmoid',name='encode_layer4')

    def call(self, denseinput1):
        denseoutput1=self.layer1(denseinput1)
        denseoutput2=self.layer2(denseoutput1)
        denseoutput3=self.layer3(denseoutput2)
        denseoutput4=self.layer4(denseoutput3)
        low_dimension_feature.append(denseoutput4)
        
