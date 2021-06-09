import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import music21 as mc
import pickle
import os
import random
import copy
import tkinter as tk
from PIL import Image,ImageTk
import pygame as pg
import threading
import time


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


# 读取指定后缀的所有文件
# base是文件目录，suffix为指定后缀
def find_all_file(base, suffix):
    # 扫描文件目录信息
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith(suffix):
                # 找到所有后缀符合条件的文件并迭代返回
                fullname = os.path.join(root, f)
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


def save_data(s):
    # sum_notes是一个二维的列表，加入有十首歌那就十行，每行保存一首歌的所有音符
    sum_notes=[]
    # note2num 用note拿num的映射
    note2num={}
    # num2note 用num拿note的映射
    num2note={}
    # 读取midi的路径，用路径把不同的风格分隔开
    base = 'Files/'+s+'/'
    print(base)
    # 对每个midi文件进行操作
    for file in find_all_file(base,'.mid'):
        # 将多余的信息剔除，得到midi文件的名字
        filenames=file.split('/')
        filename=filenames[len(filenames)-1]
        filename=filename.rstrip('.mid')

        # 更新Text界面上的信息
        textvar='生成'+filename+'\n'
        print(textvar)
        # text.insert('insert',textvar)
        # text.update()

        notes = read_midi_file(file)
        for note in notes:
            sum_notes.append(note)
        if len(notes)>0:
            if not os.path.exists(s):
                os.mkdir(s)
            with open(s+'/'+filename+'.bin', 'wb') as f:
                pickle.dump(notes, f)
    notes_member=sorted(set(sum_notes))
    for i in range(len(notes_member)):
        note2num[notes_member[i]]=i
        num2note[str(i)]=notes_member[i]
    print(note2num)
    if not os.path.exists(s):
        os.mkdir(s)
    with open(s+'/'+'note2num'+'.bin', 'wb') as f:
        pickle.dump(note2num, f)
    with open(s+'/'+'num2note'+'.bin', 'wb') as f:
        pickle.dump(num2note, f)


def load_data(s):
    base = s+'/'
    notes=[]
    sum_notes=[]
    note2num={}
    num2note={}
    for file in find_all_file(base,'.bin'):
        if file!=s+'/note2num.bin' and file!=s+'/num2note.bin':
            with open(file,'rb') as f:
                note_group=pickle.load(f)
                notes.append(note_group)
                for note in note_group:
                    sum_notes.append(note)
#     notes_member=sorted(set(sum_notes))
#     for i in range(len(notes_member)):
#         note2num[notes_member[i]]=i
#         num2note[str(i)]=notes_member[i]
    print('6666'+s+'/note2num.bin')
    with open(s+'/note2num.bin','rb') as f:
        note2num=pickle.load(f)
    with open(s+'/num2note.bin','rb') as f:
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


def train(file=None,style=None):
    notes, note2num, num2note = load_data(style)
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


def generate_chopin(choice):
    # 确定每多少个音符作为输入信息
    predict_length = 128
    # 加载模型
    file = 'model/chopin.hdf5'
    # 加载数据集和两个映射
    notes, note2num, num2note = load_data('chopin')
    # 神经网络加载指定的训练完成的模型
    model = train(file,'chopin')
    # notes_len=len(notes)
    # choice=int(random.random()*notes_len)
    # 选择某首歌作为引子
    print(choice)
    list_x = read_midi_file('Files/chopin/'+choice)

    # 选择全空做引子
    #     list_x=[]
    #     for i in range(128):
    #         list_x.append(' ')
    # 这里将原本的字符串利用之前加载的映射全部转为数字
    for i in range(len(list_x)):
        list_x[i] = note2num[list_x[i]]
    # 输入的起点选择未被训练过的128个音符
    # 将真正的输入数据导入到test_x中
    start = len(list_x)-128-1
    test_x = []
    res = []
    for i in range(start, start + predict_length):
        print(i)
        test_x.append(list_x[i])

    # 让AI去生成300个音符
    for i in range(300):
        # 信息在Text上显示
        text.insert('insert','chopin风格生成第'+str(i+1)+'个音符\n')
        text.update()

        # 先复制一份原来的输入
        input_x = copy.deepcopy(test_x)
        # reshape一下，不然无法进神经网络里
        input_x = np.reshape(input_x, (1, len(input_x), 1))
        # 正则化
        input_x = input_x / float(len(num2note))
        # 让AI去预测下一个音符
        test_y = model.predict(input_x)
        new_num = np.argmax(test_y)
        new_note = num2note[str(new_num)]
        # 将新的音符推入原来的输入中作为新的输入，同时也要放入res(结果)
        res.append(new_note)
        test_x.append(new_num)
        # 将test_x的第一个输入推出，因为要保证输入始终是128
        test_x = test_x[1:]
    # 生成midi文件
    offset = 0
    music = []
    # 生成 Note（音符）或 Chord（和弦）对象
    for note in res:
        # 将音符和持续时间作切分
        note, duration = note.split('|')
        # 如果是和弦的话还要继续切分
        if '^' in note or note.isdigit():
            chord_list = note.split('^')
            music_notes = []
            # 和弦是由多个音符同时按下构成的，要一个个去处理
            for i in chord_list:
                music_note = mc.note.Note(int(i))
                music_note.duration = mc.duration.Duration(float(duration))
                music_note.storedInstrument = mc.instrument.Piano()  # 乐器用钢琴 (piano)
                music_notes.append(music_note)
            # 最后构建出和弦对象
            chord = mc.chord.Chord(music_notes)
            chord.offset = offset
            music.append(chord)
        # 是停顿符rest
        elif note == ' ':
            music_note = mc.note.Rest()
            music_note.offset = offset
            # 停顿也有duration，这一点非常关键
            music_note.duration = mc.duration.Duration(float(duration))
            music.append(music_note)
        # 是 Note
        else:
            music_note = mc.note.Note(note)
            music_note.offset = offset
            music_note.duration = mc.duration.Duration(float(duration))
            music_note.storedInstrument = mc.instrument.Piano()
            music.append(music_note)

        # 每次迭代都将偏移增加，这里可以控制速度，偏移越小歌曲速度越快
        offset += 0.43

    # 创建音乐流（Stream）
    res_stream = mc.stream.Stream(music)

    #处理midi文件的路径，可以找到midi文件的名字
    path_list = choice.split('/')
    path = path_list[-1]

    # 写入midi文件
    res_stream.write('midi', fp='生成音乐/chopin.mid')

    # 显示生成完成
    text.insert('insert', 'chopin风格生成完成\n')
    text.update()


def generate_mozart(choice):
    # 确定每多少个音符作为输入信息
    predict_length = 128
    # 加载模型
    file = 'model/mozart.hdf5'
    # 加载数据集和两个映射
    notes, note2num, num2note = load_data('mozart')
    # 神经网络加载指定的训练完成的模型
    model = train(file,'mozart')
    # notes_len=len(notes)
    # choice=int(random.random()*notes_len)
    # 选择某首歌作为引子
    list_x = read_midi_file('Files/mozart/'+choice)

    # 选择全空做引子
    #     list_x=[]
    #     for i in range(128):
    #         list_x.append(' ')
    # 这里将原本的字符串利用之前加载的映射全部转为数字
    for i in range(len(list_x)):
        list_x[i] = note2num[list_x[i]]
    # 输入的起点选择未被训练过的128个音符
    # 将真正的输入数据导入到test_x中
    start = len(list_x) - 128 - 1
    test_x = []
    res = []
    for i in range(start, start + predict_length):
        test_x.append(list_x[i])

    # 让AI去生成300个音符
    for i in range(300):
        # 信息在Text上显示
        text.insert('insert', 'mozart风格生成第' + str(i + 1) + '个音符\n')
        text.update()

        # 先复制一份原来的输入
        input_x = copy.deepcopy(test_x)
        # reshape一下，不然无法进神经网络里
        input_x = np.reshape(input_x, (1, len(input_x), 1))
        # 正则化
        input_x = input_x / float(len(num2note))
        # 让AI去预测下一个音符
        test_y = model.predict(input_x)
        new_num = np.argmax(test_y)
        new_note = num2note[str(new_num)]
        # 将新的音符推入原来的输入中作为新的输入，同时也要放入res(结果)
        res.append(new_note)
        test_x.append(new_num)
        # 将test_x的第一个输入推出，因为要保证输入始终是128
        test_x = test_x[1:]
    # 生成midi文件
    offset = 0
    music = []
    # 生成 Note（音符）或 Chord（和弦）对象
    for note in res:
        # 将音符和持续时间作切分
        note, duration = note.split('|')
        # 如果是和弦的话还要继续切分
        if '^' in note or note.isdigit():
            chord_list = note.split('^')
            music_notes = []
            # 和弦是由多个音符同时按下构成的，要一个个去处理
            for i in chord_list:
                music_note = mc.note.Note(int(i))
                music_note.duration = mc.duration.Duration(float(duration))
                music_note.storedInstrument = mc.instrument.Piano()  # 乐器用钢琴 (piano)
                music_notes.append(music_note)
            # 最后构建出和弦对象
            chord = mc.chord.Chord(music_notes)
            chord.offset = offset
            music.append(chord)
        # 是停顿符rest
        elif note == ' ':
            music_note = mc.note.Rest()
            music_note.offset = offset
            # 停顿也有duration，这一点非常关键
            music_note.duration = mc.duration.Duration(float(duration))
            music.append(music_note)
        # 是 Note
        else:
            music_note = mc.note.Note(note)
            music_note.offset = offset
            music_note.duration = mc.duration.Duration(float(duration))
            music_note.storedInstrument = mc.instrument.Piano()
            music.append(music_note)

        # 每次迭代都将偏移增加，这里可以控制速度，偏移越小歌曲速度越快
        offset += 0.45

    # 创建音乐流（Stream）
    res_stream = mc.stream.Stream(music)

    # 处理midi文件的路径，可以找到midi文件的名字
    path_list = choice.split('/')
    path = path_list[-1]

    # 写入midi文件
    res_stream.write('midi', fp='生成音乐/mozart.mid')

    # 显示生成完成
    text.insert('insert', 'mozart风格生成完成\n')
    text.update()


def generate_jazz(choice):
    # 确定每多少个音符作为输入信息
    predict_length = 128
    # 加载模型
    file = 'model/jazz.hdf5'
    # 加载数据集和两个映射
    notes, note2num, num2note = load_data('jazz')
    # 神经网络加载指定的训练完成的模型
    model = train(file,'jazz')
    # notes_len=len(notes)
    # choice=int(random.random()*notes_len)
    # 选择某首歌作为引子
    list_x = read_midi_file('Files/jazz/'+choice)

    # 选择全空做引子
    #     list_x=[]
    #     for i in range(128):
    #         list_x.append(' ')
    # 这里将原本的字符串利用之前加载的映射全部转为数字
    for i in range(len(list_x)):
        list_x[i] = note2num[list_x[i]]
    # 输入的起点选择未被训练过的128个音符
    # 将真正的输入数据导入到test_x中
    start = len(list_x) - 128 - 1
    test_x = []
    res = []
    for i in range(start, start + predict_length):
        test_x.append(list_x[i])

    # 让AI去生成300个音符
    for i in range(300):
        # 信息在Text上显示
        text.insert('insert', 'jazz风格生成第' + str(i + 1) + '个音符\n')
        text.update()

        # 先复制一份原来的输入
        input_x = copy.deepcopy(test_x)
        # reshape一下，不然无法进神经网络里
        input_x = np.reshape(input_x, (1, len(input_x), 1))
        # 正则化
        input_x = input_x / float(len(num2note))
        # 让AI去预测下一个音符
        test_y = model.predict(input_x)
        new_num = np.argmax(test_y)
        new_note = num2note[str(new_num)]
        # 将新的音符推入原来的输入中作为新的输入，同时也要放入res(结果)
        res.append(new_note)
        test_x.append(new_num)
        # 将test_x的第一个输入推出，因为要保证输入始终是128
        test_x = test_x[1:]
    # 生成midi文件
    offset = 0
    music = []
    # 生成 Note（音符）或 Chord（和弦）对象
    for note in res:
        # 将音符和持续时间作切分
        note, duration = note.split('|')
        # 如果是和弦的话还要继续切分
        if '^' in note or note.isdigit():
            chord_list = note.split('^')
            music_notes = []
            # 和弦是由多个音符同时按下构成的，要一个个去处理
            for i in chord_list:
                music_note = mc.note.Note(int(i))
                music_note.duration = mc.duration.Duration(float(duration))
                music_note.storedInstrument = mc.instrument.Piano()  # 乐器用钢琴 (piano)
                music_notes.append(music_note)
            # 最后构建出和弦对象
            chord = mc.chord.Chord(music_notes)
            chord.offset = offset
            music.append(chord)
        # 是停顿符rest
        elif note == ' ':
            music_note = mc.note.Rest()
            music_note.offset = offset
            # 停顿也有duration，这一点非常关键
            music_note.duration = mc.duration.Duration(float(duration))
            music.append(music_note)
        # 是 Note
        else:
            music_note = mc.note.Note(note)
            music_note.offset = offset
            music_note.duration = mc.duration.Duration(float(duration))
            music_note.storedInstrument = mc.instrument.Piano()
            music.append(music_note)

        # 每次迭代都将偏移增加，这里可以控制速度，偏移越小歌曲速度越快
        offset += 0.39

    # 创建音乐流（Stream）
    res_stream = mc.stream.Stream(music)

    # 处理midi文件的路径，可以找到midi文件的名字
    path_list = choice.split('/')
    path = path_list[-1]

    # 写入midi文件
    res_stream.write('midi', fp='生成音乐/jazz.mid')

    # 显示生成完成
    text.insert('insert', 'jazz风格生成完成\n')
    text.update()


def generate_light(choice):
    # 确定每多少个音符作为输入信息
    predict_length = 128
    # 加载模型
    file = 'model/light.hdf5'
    # 加载数据集和两个映射
    notes, note2num, num2note = load_data('light')
    # 神经网络加载指定的训练完成的模型
    model = train(file,'light')
    # notes_len=len(notes)
    # choice=int(random.random()*notes_len)
    # 选择某首歌作为引子
    list_x = read_midi_file('Files/light/'+choice)

    # 选择全空做引子
    #     list_x=[]
    #     for i in range(128):
    #         list_x.append(' ')
    # 这里将原本的字符串利用之前加载的映射全部转为数字
    for i in range(len(list_x)):
        list_x[i] = note2num[list_x[i]]
    # 输入的起点选择未被训练过的128个音符
    # 将真正的输入数据导入到test_x中
    start = len(list_x) - 128 - 1
    test_x = []
    res = []
    for i in range(start, start + predict_length):
        test_x.append(list_x[i])

    # 让AI去生成300个音符
    for i in range(300):
        # 信息在Text上显示
        text.insert('insert', 'light风格生成第' + str(i + 1) + '个音符\n')
        text.update()

        # 先复制一份原来的输入
        input_x = copy.deepcopy(test_x)
        # reshape一下，不然无法进神经网络里
        input_x = np.reshape(input_x, (1, len(input_x), 1))
        # 正则化
        input_x = input_x / float(len(num2note))
        # 让AI去预测下一个音符
        test_y = model.predict(input_x)
        new_num = np.argmax(test_y)
        new_note = num2note[str(new_num)]
        # 将新的音符推入原来的输入中作为新的输入，同时也要放入res(结果)
        res.append(new_note)
        test_x.append(new_num)
        # 将test_x的第一个输入推出，因为要保证输入始终是128
        test_x = test_x[1:]
    # 生成midi文件
    offset = 0
    music = []
    # 生成 Note（音符）或 Chord（和弦）对象
    for note in res:
        # 将音符和持续时间作切分
        note, duration = note.split('|')
        # 如果是和弦的话还要继续切分
        if '^' in note or note.isdigit():
            chord_list = note.split('^')
            music_notes = []
            # 和弦是由多个音符同时按下构成的，要一个个去处理
            for i in chord_list:
                music_note = mc.note.Note(int(i))
                music_note.duration = mc.duration.Duration(float(duration))
                music_note.storedInstrument = mc.instrument.Piano()  # 乐器用钢琴 (piano)
                music_notes.append(music_note)
            # 最后构建出和弦对象
            chord = mc.chord.Chord(music_notes)
            chord.offset = offset
            music.append(chord)
        # 是停顿符rest
        elif note == ' ':
            music_note = mc.note.Rest()
            music_note.offset = offset
            # 停顿也有duration，这一点非常关键
            music_note.duration = mc.duration.Duration(float(duration))
            music.append(music_note)
        # 是 Note
        else:
            music_note = mc.note.Note(note)
            music_note.offset = offset
            music_note.duration = mc.duration.Duration(float(duration))
            music_note.storedInstrument = mc.instrument.Piano()
            music.append(music_note)

        # 每次迭代都将偏移增加，这里可以控制速度，偏移越小歌曲速度越快
        offset += 0.5

    # 创建音乐流（Stream）
    res_stream = mc.stream.Stream(music)

    # 处理midi文件的路径，可以找到midi文件的名字
    path_list = choice.split('/')
    path = path_list[-1]

    # 写入midi文件
    res_stream.write('midi', fp='生成音乐/light.mid')

    # 显示生成完成
    text.insert('insert', 'light风格生成完成\n')
    text.update()

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


def play_music(music_file):
  '''
  stream music with mixer.music module in blocking manner
  this will stream the sound from disk while playing
  '''
  clock = pg.time.Clock()
  try:
    pg.mixer.music.load(music_file)
    print("Music file {} loaded!".format(music_file))
  except pygame.error:
    print("File {} not found! {}".format(music_file, pg.get_error()))
    return
  pg.mixer.music.play()
  # check if playback has finished
  while pg.mixer.music.get_busy():
    clock.tick(30)


def play_chopin():
    # pick a midi or MP3 music file you have in the working folder
    # or give full pathname
    music_file = "生成音乐/chopin.mid"
    # music_file = "Drumtrack.mp3"
    freq = 44100  # audio CD quality
    bitsize = -16  # unsigned 16 bit
    channels = 2  # 1 is mono, 2 is stereo
    buffer = 2048  # number of samples (experiment to get right sound)
    pg.mixer.init(freq, bitsize, channels, buffer)
    # optional volume 0 to 1.0
    pg.mixer.music.set_volume(1)
    try:
        play_music(music_file)
    except KeyboardInterrupt:
        # if user hits Ctrl/C then exit
        # (works only in console mode)
        pg.mixer.music.fadeout(1000)
        pg.mixer.music.stop()
        raise SystemExit

def play_mozart():
    # pick a midi or MP3 music file you have in the working folder
    # or give full pathname
    music_file = "生成音乐/mozart.mid"
    # music_file = "Drumtrack.mp3"
    freq = 44100  # audio CD quality
    bitsize = -16  # unsigned 16 bit
    channels = 2  # 1 is mono, 2 is stereo
    buffer = 2048  # number of samples (experiment to get right sound)
    pg.mixer.init(freq, bitsize, channels, buffer)
    # optional volume 0 to 1.0
    pg.mixer.music.set_volume(1)
    try:
        play_music(music_file)
    except KeyboardInterrupt:
        # if user hits Ctrl/C then exit
        # (works only in console mode)
        pg.mixer.music.fadeout(1000)
        pg.mixer.music.stop()
        raise SystemExit

def play_jazz():
    # pick a midi or MP3 music file you have in the working folder
    # or give full pathname
    music_file = "生成音乐/jazz.mid"
    # music_file = "Drumtrack.mp3"
    freq = 44100  # audio CD quality
    bitsize = -16  # unsigned 16 bit
    channels = 2  # 1 is mono, 2 is stereo
    buffer = 2048  # number of samples (experiment to get right sound)
    pg.mixer.init(freq, bitsize, channels, buffer)
    # optional volume 0 to 1.0
    pg.mixer.music.set_volume(1)
    try:
        play_music(music_file)
    except KeyboardInterrupt:
        # if user hits Ctrl/C then exit
        # (works only in console mode)
        pg.mixer.music.fadeout(1000)
        pg.mixer.music.stop()
        raise SystemExit

def play_light():
    # pick a midi or MP3 music file you have in the working folder
    # or give full pathname
    music_file = "生成音乐/light.mid"
    # music_file = "Drumtrack.mp3"
    freq = 44100  # audio CD quality
    bitsize = -16  # unsigned 16 bit
    channels = 2  # 1 is mono, 2 is stereo
    buffer = 2048  # number of samples (experiment to get right sound)
    pg.mixer.init(freq, bitsize, channels, buffer)
    # optional volume 0 to 1.0
    pg.mixer.music.set_volume(1)
    try:
        play_music(music_file)
    except KeyboardInterrupt:
        # if user hits Ctrl/C then exit
        # (works only in console mode)
        pg.mixer.music.fadeout(1000)
        pg.mixer.music.stop()
        raise SystemExit


# 使用线程类来同时生成和读取数据
class myThread(threading.Thread):
    # s即为要读取哪个文件夹(chopin or mozart or jazz or light)
    def __init__(self,s):
        threading.Thread.__init__(self)
        self.path = s
    def run(self):
        # 输出线程打开的信息
        textvar="开始线程："+self.path+'\n'
        print(textvar)
        # text.insert('insert',textvar)
        # text.update()

        # 读取数据
        save_data(self.path)

        # 输出线程退出的信息
        textvar="退出线程："+self.path+'\n'
        print(textvar)
        # text.insert('insert',textvar)
        # text.update()

def prepare():
    # 将Files里的chopin作品加载为bin文件
    thread_chopin=myThread('chopin')
    # 将Files里的mozart作品加载为bin文件
    thread_mozart=myThread('mozart')
    # 将Files里的jazz作品加载为bin文件
    thread_jazz=myThread('jazz')
    # 将Files里的轻音乐作品加载为bin文件
    thread_light=myThread('light')

    # 先全部start
    thread_chopin.start()
    thread_mozart.start()
    thread_jazz.start()
    thread_light.start()

    # 显示信息
    text.insert('insert','开始读取\n')
    text.update()

    # 全部join开始运行
    thread_chopin.join()
    thread_mozart.join()
    thread_jazz.join()
    thread_light.join()

    # 显示信息
    text.insert('insert', '全部读取结束\n')
    text.update()

def prepare_test():
    # 输出线程打开的信息
    textvar = "开始线程："  + '\n'
    text.insert('insert', textvar)
    text.update()

    # 读取数据
    # save_data(path)

    # 输出线程退出的信息
    textvar = "退出线程："  + '\n'
    text.insert('insert', textvar)
    text.update()

def generate():
    generate_chopin('Waltzes--Op. 70. No. 2. in Ab --Chopin.mid')
    generate_mozart('Piano Sonata No. 7. in C,[K309]--Allegro --Mozart.mid')
    generate_jazz('after_the_love_has_gone_(earth!_wind_!_fire)[1].mid')
    generate_light('超好听-Again-アゲイン-横山克-四月是你的谎言.mid')

# 将图片适配成适合窗口的大小
def get_image(path,w,h):
    img=Image.open(path).resize((w,h))
    return ImageTk.PhotoImage(img)

frame = tk.Tk()
# 定义界面大小
frame.geometry('800x600')
frame.resizable(False,False)
# 界面标题
frame.title('AI多风格音乐生成器')


# 创建画布
canvas=tk.Canvas(frame,width=800,height=600)
# 创建图片对象
img=get_image('background.png',800,600)
# 在画布上显示出图片
canvas.create_image(400,300,image=img)
# 显示画布
canvas.pack()


# 不同风格音乐播放的按钮
tk.Button(frame,text='古典(肖邦)',command=play_chopin,width=8,bg='blue').place(x=120, y=110)
tk.Button(frame,text='古典(莫扎特)',command=play_mozart,width=8,bg='blue').place(x=120, y=230)
tk.Button(frame,text='爵士',command=play_jazz,width=8,bg='blue').place(x=120,y=350)
tk.Button(frame,text='轻音乐',command=play_light,width=8,bg='blue').place(x=120,y=470)

# 读取midi文件加载输入数据的按钮
tk.Button(frame,text='加载数据',command=prepare,width=10).place(x=260,y=10)

# 随机生成所有风格音乐的按钮
tk.Button(frame,text='生成音乐',command=generate,width=10).place(x=420,y=10)

# 显示信息的文本框
text=tk.Text(frame,height=30,width=50)
text.place(x=400,y=80)


frame.mainloop()