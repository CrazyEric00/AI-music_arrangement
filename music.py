import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import music21 as mc
import pickle
import os
import random
import copy


class myLSTM(tf.keras.layers.Layer):
    def __init__(self,output_size):
        # 使用最基础的Layer进行初始化
        super(myLSTM,self).__init__()
        # 输出的长度
        self.output_size=output_size

    def build(self,input_shape):
        super(myLSTM,self).__init__()
        input_size=int(input_shape[-1])

        self.wf=self.add_weight('wf',shape=(input_size,self.output_size))
        self.wi = self.add_weight('wi', shape=(input_size, self.output_size))
        self.wo = self.add_weight('wo', shape=(input_size, self.output_size))
        self.wc = self.add_weight('wc', shape=(input_size, self.output_size))

        self.uf=self.add_weight('uf',shape=(self.output_size,self.output_size))
        self.ui = self.add_weight('ui', shape=(self.output_size, self.output_size))
        self.uo = self.add_weight('uo', shape=(self.output_size, self.output_size))
        self.uc = self.add_weight('uc', shape=(self.output_size, self.output_size))

        self.bf = self.add_weight('bf', shape=(1, self.output_size))
        self.bi = self.add_weight('bi', shape=(1, self.output_size))
        self.bo = self.add_weight('bo', shape=(1, self.output_size))
        self.bc = self.add_weight('bc', shape=(1, self.output_size))


#读取指定后缀的所有文件
#base是文件目录，suffix为指定后缀
def find_all_file(base,suffix):
    #扫描文件目录信息
    for root,ds,fs in os.walk(base):
        for f in fs:
            if f.endswith(suffix):
                #找到所有后缀符合条件的文件并迭代返回
                fullname=os.path.join(root,f)
                yield fullname


# 读取单个midi文件并返回notes
def read_midi_file(file):
    notes = []
    notes_merge_rest = []
    print('读取文件:' + file)
    try:
        stream = mc.converter.parse(file)
        instruments = mc.instrument.partitionByInstrument(stream)
        for instrument in instruments.parts:
            print(str(instrument))
            if 'Piano' in str(instrument):
                elements = instrument.recurse()
                for element in elements:
                    if isinstance(element, mc.note.Note):
                        # 机智的操作要来了
                        ###############
                        if element.duration.quarterLength > 1:
                            element.duration.quarterLength = int(element.duration.quarterLength)
                        elif element.duration.quarterLength > 0.3 and element.duration.quarterLength < 0.4:
                            element.duration.quarterLength = 0.25
                        elif element.duration.quarterLength > 0.6 and element.duration.quarterLength < 0.7:
                            element.duration.quarterLength = 0.75
                        if element.duration.quarterLength > 4.0:
                            element.duration.quarterLength = 4.0
                        ###############
                        notes.append(str(element.pitch) + '|' + str(float(element.duration.quarterLength)))
                    elif isinstance(element, mc.chord.Chord):
                        s = '^'.join(str(i) for i in element.normalOrder)
                        # 机智的操作要来了
                        ###############
                        if element.duration.quarterLength > 1:
                            element.duration.quarterLength = int(element.duration.quarterLength)
                        elif element.duration.quarterLength > 0.3 and element.duration.quarterLength < 0.4:
                            element.duration.quarterLength = 0.25
                        elif element.duration.quarterLength > 0.6 and element.duration.quarterLength < 0.7:
                            element.duration.quarterLength = 0.75
                        if element.duration.quarterLength > 4.0:
                            element.duration.quarterLength = 4.0
                        ###############
                        notes.append(s + '|' + str(float(element.duration.quarterLength)))
                    elif isinstance(element, mc.note.Rest):
                        if notes:
                            # 机智的操作要来了
                            ###############
                            if element.duration.quarterLength > 1:
                                element.duration.quarterLength = int(element.duration.quarterLength)
                            elif element.duration.quarterLength > 0.3 and element.duration.quarterLength < 0.4:
                                element.duration.quarterLength = 0.25
                            elif element.duration.quarterLength > 0.6 and element.duration.quarterLength < 0.7:
                                element.duration.quarterLength = 0.75
                            if element.duration.quarterLength > 4.0:
                                element.duration.quarterLength = 4.0
                            ###############
                            # 已经存在音符才放入休止，防止开头的一堆休止符号
                            notes.append(' ' + '|' + str(float(element.duration.quarterLength)))
        # 处理Rest，合并冗余的rest，防止模型产生rest的几率飙升
        i = 0
        while (i < len(notes)):
            a, b = notes[i].split('|')
            if a == ' ':
                j = i + 1
                duration = float(b)
                while (j < len(notes)):
                    c, d = notes[j].split('|')
                    if c == ' ':
                        duration += float(d)
                        j += 1
                    else:
                        i = j - 1
                        # 机智的操作要来了
                        ###############
                        if duration > 1:
                            duration = int(duration)
                        elif duration > 0.3 and duration < 0.4:
                            duration = 0.25
                        elif duration > 0.6 and duration < 0.7:
                            duration = 0.75
                        if duration > 4.0:
                            duration = 4.0
                        ###############
                        notes_merge_rest.append(' |' + str(float(duration)))
                        break
            else:
                notes_merge_rest.append(notes[i])
            i += 1
    except Exception:
        print('出错:' + file)
        return notes_merge_rest

    return notes_merge_rest

def save_data():
    sum_notes=[]
    note2num={}
    num2note={}
    base = 'Files/mozart/'
    for file in find_all_file(base,'.mid'):
        filenames=file.split('/')
        filename=filenames[len(filenames)-1]
        filename=filename.rstrip('.mid')
        notes = read_midi_file(file)
        for note in notes:
            sum_notes.append(note)
        if len(notes)>0:
            if not os.path.exists("data"):
                os.mkdir("data")
            with open('data/'+filename+'.bin', 'wb') as f:
                pickle.dump(notes, f)
    notes_member=sorted(set(sum_notes))
    for i in range(len(notes_member)):
        note2num[notes_member[i]]=i
        num2note[str(i)]=notes_member[i]
    print(note2num)
    if not os.path.exists("data"):
        os.mkdir("data")
    with open('data/'+'note2num'+'.bin', 'wb') as f:
        pickle.dump(note2num, f)
    with open('data/'+'num2note'+'.bin', 'wb') as f:
        pickle.dump(num2note, f)

def load_data():
    base = 'data/'
    notes=[]
    sum_notes=[]
    note2num={}
    num2note={}
    for file in find_all_file(base,'.bin'):
        if file!='data/note2num.bin' and file!='data/num2note.bin':
            with open(file,'rb') as f:
                note_group=pickle.load(f)
                notes.append(note_group)
                for note in note_group:
                    sum_notes.append(note)
#     notes_member=sorted(set(sum_notes))
#     for i in range(len(notes_member)):
#         note2num[notes_member[i]]=i
#         num2note[str(i)]=notes_member[i]
    with open('data/note2num.bin','rb') as f:
        note2num=pickle.load(f)
    with open('data/num2note.bin','rb') as f:
        num2note=pickle.load(f)
    return notes,note2num,num2note


def data_processing(notes,note2num,num2note):
    predict_length=128
    train_x=[]
    train_y=[]
    for note_group in notes:
        for i in range(len(note_group)-predict_length-1):
            note_list=note_group[i:i+predict_length]
            for j in range(len(note_list)):
                note_list[j]=note2num[note_list[j]]
            train_x.append(note_list)
            train_y.append(note2num[note_group[i+predict_length]])
    n=len(train_x)
    train_x=np.reshape(train_x,(n,predict_length,1))
    train_x=train_x/float(len(note2num))
    train_y=tf.keras.utils.to_categorical(train_y)
    return train_x,train_y


def train(file=None):
    notes, note2num, num2note = load_data()
    train_x, train_y = data_processing(notes, note2num, num2note)
    print(train_x.shape)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(
        256,
        input_shape=(train_x.shape[1], train_x.shape[2]),
        return_sequences=True  # 返回所有的输出序列（Sequences）
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.LSTM(256, return_sequences=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(len(note2num)))
    model.add(tf.keras.layers.Activation('softmax'))
    # 计算误差（先用 Softmax 计算百分比概率，再用 Cross entropy（交叉熵）来计算百分比概率和对应的独热码之间的误差）
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    filepath = "model/weights.{epoch:02d}-{loss:.4f}.hdf5"

    # 用 Checkpoint（检查点）文件在每一个 Epoch 结束时保存模型的参数（Weights）
    # 不怕训练过程中丢失模型参数。可以在我们对 Loss（损失）满意了的时候随时停止训练
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath,  # 保存的文件路径
        monitor='loss',  # 监控的对象是 损失（loss）
        verbose=0,
        period=50
    )
    callbacks_list = [checkpoint]
    if file is not None:
        model.load_weights(file)
    else:
        # model.load_weights('model/weights.100-3.7315.hdf5')
        # 用 fit 方法来训练模型
        print(model.summary())
        model.fit(train_x, train_y, epochs=200, batch_size=128, callbacks=callbacks_list)
    return model


def generate(choice):
    predict_length = 128
    file = 'model/chopin.hdf5'
    notes, note2num, num2note = load_data()
    model = train(file)
    # notes_len=len(notes)
    # choice=int(random.random()*notes_len)
    # 选择某首歌作为引子
    list_x = read_midi_file(choice)

    # 选择全空做引子
    #     list_x=[]
    #     for i in range(128):
    #         list_x.append(' ')

    for i in range(len(list_x)):
        list_x[i] = note2num[list_x[i]]
    start = len(list_x) - 128 - 1
    test_x = []
    res = []
    for i in range(start, start + predict_length):
        test_x.append(list_x[i])
    for i in range(300):
        print(i, end=' ')
        input_x = copy.deepcopy(test_x)
        input_x = np.reshape(input_x, (1, len(input_x), 1))
        input_x = input_x / float(len(num2note))
        test_y = model.predict(input_x)
        new_num = np.argmax(test_y)
        new_note = num2note[str(new_num)]
        res.append(new_note)
        test_x.append(new_num)
        test_x = test_x[1:]
    # 生成midi文件
    offset = 0
    music = []
    # 生成 Note（音符）或 Chord（和弦）对象
    for note in res:
        note, duration = note.split('|')
        if '^' in note or note.isdigit():
            chord_list = note.split('^')
            music_notes = []
            for i in chord_list:
                music_note = mc.note.Note(int(i))
                music_note.duration = mc.duration.Duration(float(duration))
                music_note.storedInstrument = mc.instrument.Piano()  # 乐器用钢琴 (piano)
                music_notes.append(music_note)
            chord = mc.chord.Chord(music_notes)
            chord.offset = offset
            music.append(chord)
        # 是停顿符rest
        elif note == ' ':
            music_note = mc.note.Rest()
            music_note.offset = offset
            music_note.duration = mc.duration.Duration(float(duration))
            music.append(music_note)
        # 是 Note
        else:
            music_note = mc.note.Note(note)
            music_note.offset = offset
            music_note.duration = mc.duration.Duration(float(duration))
            music_note.storedInstrument = mc.instrument.Piano()
            music.append(music_note)

        # 每次迭代都将偏移增加，这样才不会交叠覆盖
        offset += 0.5

    # 创建音乐流（Stream）
    res_stream = mc.stream.Stream(music)

    path_list = choice.split('/')
    path = path_list[-1]

    # 写入 MIDI 文件
    res_stream.write('midi', fp='生成音乐/' + path)


def test_generate(choice):
    notes, note2num, num2note = load_data()
    res = read_midi_file(choice)
    # 生成midi文件
    offset = 0
    music = []
    # 生成 Note（音符）或 Chord（和弦）对象
    for note in res:
        # 休止符直接生成不设置duration
        if note == ' ':
            music_note = mc.note.Rest()
            music_note.offset = offset
            music.append(music_note)
        # 是 Chord。格式例如： 4^15^7
        else:
            note, duration = note.split('|')
            if '^' in note:
                chord_list = note.split('^')
                music_notes = []
                for i in chord_list:
                    music_note = mc.note.Note(int(i))
                    music_note.duration = mc.duration.Duration(duration)
                    music_note.storedInstrument = mc.instrument.Piano()  # 乐器用钢琴 (piano)
                    music_notes.append(music_note)
                chord = mc.chord.Chord(music_notes)
                chord.offset = offset
                music.append(chord)
            # 是 Note
            else:
                music_note = mc.note.Note(note)
                music_note.offset = offset
                music_note.duration = mc.duration.Duration(duration)
                music_note.storedInstrument = mc.instrument.Piano()
                music.append(music_note)

        # 每次迭代都将偏移增加，这样才不会交叠覆盖
        offset += 0.5
    # 创建音乐流（Stream）
    res_stream = mc.stream.Stream(music)

    # 写入 MIDI 文件
    res_stream.write('midi', fp='example.mid')

