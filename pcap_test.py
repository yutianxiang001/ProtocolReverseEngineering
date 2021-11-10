#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import activations
from tensorflow.keras.layers import Layer, Input, Embedding, LSTM, Dense, Attention, Bidirectional, SimpleRNN, GRU
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers.recurrent import RNN
from tensorflow.python.keras.regularizers import Regularizer
from tensorflow.python.ops.gen_dataset_ops import DeleteMultiDeviceIterator
from pcap import Pcap
from rtmp import RTMP
import csv
import pandas

app_text=[]
app_number=[]
app_low_dimensional_features=[]
maxlen=256
output_path='data/app-number.tsv'
output_path2='data/app-low-dimensional-features.tsv'

def callback_rtmp(ins, msg, fr):
    if msg.__class__.__name__ == "str":
        print(msg)
        if msg=="S1":
            print("s1 time:%d"%(ins.s1.time))


class PcapTest():
    """测试"""

    def test_rtmp(self):
        t = 3
        #print("tt:%s" % bin(t >> 6))

    def test_load(self):
        _pcap = Pcap()
        _gen = _pcap.parse("data/ftp1000"
                           ".pcap")
        file = open('keyInfo.txt','w')
        for _packet in _gen:
            _mac = _packet.data
            _net = _mac.data
            _trans = _net.data
            if _trans.__class__.__name__ == "TCP":
                _app = _trans.data
                if _app is not None:
                    """ print(_app)
                    print(list(_app)) """
                    app_text.append(list(_app))
                    file.write(str(_packet.head)+'\n')
                    file.write(str(_mac)+'\n')
                    file.write(str(_net)+'\n')
                    file.write(str(_trans)+'\n')
                    """ print(_packet.head)
                    print(_mac)
                    print(_net)
                    print(_trans)

                    if RTMP.find(_trans, callback_rtmp):
                        # 依次打印网络层、传输层头部
                        print("") """

        file.close()

class Encoder(keras.Model):
    def __init__(self):
        super(Encoder,self).__init__()

        self.layer1=Dense(128,activation='relu',name='encode_layer1',kernel_regularizer='L1')
        self.layer2=Dense(64,activation='sigmoid',name='encode_layer2',kernel_regularizer='L1')
        self.layer3=Dense(32,activation='relu',name='encode_layer3',kernel_regularizer='L1')
        self.layer4=Dense(2,activation='sigmoid',name='encode_layer4',kernel_regularizer='L1')

    def call(self, denseinput1):
        denseoutput1=self.layer1(denseinput1)
        denseoutput2=self.layer2(denseoutput1)
        denseoutput3=self.layer3(denseoutput2)
        denseoutput4=self.layer4(denseoutput3)
        return denseoutput4

class Decoder(keras.Model):
    def __init__(self):
        super(Decoder,self).__init__()

        self.layer1=Dense(32,activation='relu',name='decode_layer1',kernel_regularizer='L1')
        self.layer2=Dense(64,activation='tanh',name='decode_layer2',kernel_regularizer='L1')
        self.layer3=Dense(128,activation='sigmoid',name='decode_layer3',kernel_regularizer='L1')
        self.layer4=Dense(256,activation='relu',name='decode_layer4',kernel_regularizer='L1')

    def call(self, denseinput1):
        denseoutput1=self.layer1(denseinput1)
        denseoutput2=self.layer2(denseoutput1)
        denseoutput3=self.layer3(denseoutput2)
        denseoutput4=self.layer4(denseoutput3)
        return denseoutput4

def model_autoencoder():
    encoder_inputs = Input(shape=(maxlen,), name="encode_input")
    # Encoder Layer
    encoder = Encoder()
    enc_outputs = encoder(encoder_inputs)
    # Decoder Layer
    decoder = Decoder()
    dec_outputs = decoder(enc_outputs)
  
    # auto encoder model
    model = Model(inputs=encoder_inputs, outputs=dec_outputs)

    return model

def write_to_tsv(output_path: str, file_columns: list, data: list):
    csv.register_dialect('tsv_dialect', delimiter='\t', quoting=csv.QUOTE_ALL)
    with open(output_path, "w", newline="") as wf:
        writer = csv.DictWriter(wf, fieldnames=file_columns, dialect='tsv_dialect')
        writer.writerows(data)
    csv.unregister_dialect('tsv_dialect')

def write_to_tsv2(output_path: str, data: list):
    with open(output_path,'w') as f:
        tsv_w=csv.writer(f,delimiter='\t')
        for line in data:
            tsv_w.writerow(line)

if __name__ == "__main__":


    #实例化pcap分析相关类，list函数效果竟然这么好能直接数字化也是奇了。那就直接导进自编码器中
    pcap_analyzer=PcapTest()
    pcap_analyzer.test_load()
    app_number=app_text

    #过长应用层报文切割
    """ print(app_number[:1])
    print(len(app_number)) """
    app_number = keras.preprocessing.sequence.pad_sequences(app_number, padding='post', maxlen=256, truncating='post', value=0)
    print(len(app_number))
    print(app_number[:5])
    print([str(x) for x in list(range(maxlen))])
    #write_to_tsv(output_path=output_path,file_columns=[str(x) for x in list(range(maxlen))],data=app_number)
    write_to_tsv2(output_path=output_path,data=app_number)

    model=model_autoencoder()
    model.summary()
    loss_fn = keras.losses.MeanSquaredError()
    model.compile(loss=loss_fn, optimizer='adam')
    model.fit(app_number,app_number,batch_size=16,epochs=300)
    predictor=Model(inputs=model.input, outputs=model.get_layer('encoder').output)
    predictor.summary()
    app_low_dimensional_features=predictor.predict(app_number)
    print(len(app_low_dimensional_features))
    print(app_low_dimensional_features)
    print(app_low_dimensional_features[:50])
    write_to_tsv2(output_path=output_path2,data=app_low_dimensional_features)








