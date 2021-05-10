#-*- coding = utf-8 -*-
import os
import wave
import numpy as np
import matplotlib.pyplot as plt
import librosa
import shutil
import pickle

'''
source_path = "D:/DECASE_data/audio"
target_path = "D:/decaset1B/audio"
if not os.path.exists(target_path):     #目标文件夹不存在就新建
    os.makedirs(target_path)
if os.path.exists(source_path):
    for root, dirs, wavefiles in os.walk(source_path):
        for file in wavefiles:
            wav_file = os.path.join(root, file)
            shutil.copy(wav_file, target_path)
            #print(wav_file)
'''


#提取特征
savepath = "./log_mel_features_v2"
if not os.path.exists(savepath):     #目标文件夹不存在就新建
    os.makedirs(savepath)

wavelist=[]
files = os.listdir("./audio")
for filename in files:
    wavelist.append(filename)


for wav in wavelist:
    audioname = wav.split('.wav')[0]
    wavepath = os.path.join("./audio/", wav)
    resample_fs = 32000
    audiodata, fs = librosa.core.load(wavepath, sr=resample_fs)
    fft_window_size = 1024 # int(np.floor(1024 * (resample_fs / resample_fs)))
    fft_hop_size = 320 # int(np.floor(320 * (resample_fs / resample_fs)))
    # 1024/32000=0.032, 32ms一帧，hop=16ms
    window = 'hann'
    mel_bins = 64
    # print(wavepath, audiodata.shape, fs)
    log_offset = 0.01  # Offset used for stabilized log of input mel-spectrogram.

    # S为频谱,短时傅里叶变换
    S = np.abs(librosa.stft(y=audiodata,
                            n_fft=fft_window_size,
                            hop_length=fft_hop_size,
                            center=True,
                            window=window,
                            pad_mode='reflect')) ** 2

    # print(mel_basis.shape)  # (64, 1025=1024+1)
    fmax = int(fs / 2)
    fmin = 50
    melW = librosa.filters.mel(sr=fs, n_fft=fft_window_size,
                               n_mels=mel_bins, fmin=50, fmax=fmax)
    mel_S = np.dot(melW, S).T  # mel_S是梅尔频谱
    
    log_mel_S = np.log(mel_spectrogram + log_offset)
    
    print(log_mel_S.shape)

    audiopath = os.path.join(savepath,  audioname+ '.pickle')
    pickle.dump(log_mel_S, open(audiopath, 'wb'))  # pickle.HIGHEST_PROTOCOL
else:
    print('done:', savepath)
