import torch
import numpy as np
import argparse
import scipy.io
from scipy.io import loadmat as loadmat
from scipy.io import savemat as savemat
import scipy.io.wavfile as wav
from scipy.io import wavfile
import scipy
#import pyfftw
import math
import subprocess
from features import melfcc, rastaplp
from signal_extras import fftfilt, gammatone, cochleagram, synthesis
from scipy.io import wavfile
from random import randint
import sys
from scipy import signal
import argparse
import random
import os
import librosa
from utils.hparams import HParam
from utils.audio import Audio
from stft import stft
from stft import istft

def list2file(filename,my_list):
    with open(filename, 'w') as f:
        for item in my_list:
            f.write("%s\n" % item)
    return 1

def readwavfile(filename):
    fs, data = wavfile.read(filename)
    data.astype(np.float32)
    sig = data/32768.0
    # norm signal, look at how Deliang normalize!!!
    sig = 0.7*sig/np.max(sig)
    return sig, fs
def writewavfile(filename,sig,fs):
    # sig should be between -1 to 1
    wavfile.write(filename,fs,sig)
    return 1

def readlineoffile(filename):
    with open(filename) as f:
        lines = f.readlines()
    linecut = [None]*len(lines)
    for i in range(len(lines)):
        linecut[i] = lines[i][0:-1]
    return linecut
def computegammatone(sig):
    #GAMMATONE
    nc = 64
    sig = np.expand_dims(sig,axis=1)
    winLength = 320
    gm = gammatone(sig,nc)
    print(np.shape(gm))
    feat = cochleagram(gm, winLength)
    print(np.shape(feat))
    return feat

def wav2spec(x):
    if len(np.shape(x)) < 2:
        x = np.expand_dims(x, axis=1).T
    X = stft(x)
    X = np.squeeze(X, axis=0)
    mag, phase = np.abs(X), np.angle(X)
    return mag, phase

def spec2wav(mag, phase):
    mag = np.expand_dims(mag, axis=0)
    phase = np.expand_dims(phase, axis=0)
    X = mag * np.exp(1j*phase)
    x = np.squeeze(istft(X).T)
    return x

def featureextract(sig,feattype):
    nframe = len(sig) // 160
    if feattype=='main':
        constant = 5*10e6
        c = np.sqrt(constant * len(sig)/np.sum(sig ** 2))
        #print(c)
        sig = c*sig
    elif feattype=='aux':
        sig = sig
    else:
        print('need to define feature as main or aux')
    #nc = 64
    # PLP
    #rastaplp_feat = rastaplp(sig, fs=16000, win_time=0.02, hop_time=0.01, dorasta=True, modelorder=12)
    # Padding
    #zero_feat = np.zeros((rastaplp_feat.shape[0], nframe-rastaplp_feat.shape[1]), dtype=np.float32)
    #rastaplp_feat_py = np.concatenate((zero_feat, rastaplp_feat), axis=1)
    #MFCC
    #mfcc_feat = melfcc(sig, fs=16000, n_mfcc=31, n_bands=nc, window_time=0.020, hop_time=0.010, max_freq=8000)
    # Padding
    #zero_feat = np.zeros((mfcc_feat.shape[0], nframe-mfcc_feat.shape[1]), dtype=np.float32)
    #mfcc_feat_py = np.concatenate((zero_feat, mfcc_feat), axis=1)
    #GAMMATONE
    #sig = np.expand_dims(sig,axis=1)
    #winLength = 320
    #gm = gammatone(sig,nc)
    #feat = cochleagram(gm, winLength)
    #gf_feat_py = feat ** (1/15)
    # COMBINE
    #combine = np.transpose(np.concatenate((rastaplp_feat_py,mfcc_feat_py,gf_feat_py), axis=0))
    #print(np.shape(combine))
    mag , _ = wav2spec(sig)
    #del rastaplp_feat, zero_feat, mfcc_feat, mfcc_feat_py, gm, feat, gf_feat_py 
    return mag

def reading_interferingspk(tgtspk,nbspk,gender,my_list):
    auxi = [None]*(1+len(nbspk)) # 2 means that there are already 2 auxiliary -> now having only one aux from target speaker
    maxlen = 0
    y = [None]*len(nbspk)
    for i in range(len(nbspk)):
        spklist = readlineoffile('spk.list')
        if nbspk[i]==1:
            if gender=='maleonly':
                print('this inference speaker is male')
                spklist=readlineoffile('male.list')
            elif gender=='femaleonly':
                print('this inference speaker is female')
                spklist=readlineoffile('female.list')
            else:
                print('gender of the target speaker must be defined')
                exit()
        elif nbspk[i]==0:
            if gender=='maleonly':
                print('this inference speaker is female')
                spklist=readlineoffile('female.list')
            elif gender=='femaleonly':
                print('this inference speaker is male')
                spklist=readlineoffile('male.list')
            else:
                print('gender of the target speaker must be defined')
                exit()
        else:
            print('gender of the target speaker must be defined')
            exit()
        spkarray = spklist
        randnumber = randint(0,len(spkarray)-1) # -1 mean go from 0 to L-1
        ifrspk = spkarray[randnumber]
        array = ifrspk.split('/')
        inferspk = array[-1]
        while(inferspk==tgtspk):
            randnumber = randint(0,len(spkarray)-1)
            ifrspk = spkarray[randnumber]
            array = ifrspk.split('/')
            inferspk = array[-1]
        print('interfering speaker is '+inferspk)
        ifrspklist = readlineoffile('config/spk/'+inferspk+'.txt')
        randnumber = randint(0,len(ifrspklist)-1)
        ifrfile = ifrspklist[randnumber]
        print('reading audio at '+ifrfile)
        try:
            ifr, a_fs = readwavfile(ifrfile)
        except:
            print('this audio is bad, read another file...')
            randnumber = randint(0,len(ifrspklist)-1)
            ifrfile = ifrspklist[randnumber]
            print('reading audio at '+ifrfile)
            ifr, a_fs = readwavfile(ifrfile)
        # audio need to convert to 16khz    
        ifr = ifr + sys.float_info.epsilon
        y[i] = ifr/max(ifr)
        if len(y[i])>maxlen:
            maxlen = len(y[i])
        #print(str(maxlen)+"  "+str(len(y[i])))
        # read auxiliary audio
        randnumber = randint(0,len(ifrspklist)-1)
        ifrfile = ifrspklist[randnumber]
        print('reading auxiliary audio at '+ifrfile)
        try:
            auxifr, a_fs = readwavfile(ifrfile)
        except:
            print('this audio is bad, read another file...')
            randnumber = randint(0,len(ifrspklist)-1)
            ifrfile = ifrspklist[randnumber]
            print('reading audio at '+ifrfile)
            auxifr, a_fs = readwavfile(ifrfile)
        my_list.append(ifrfile)
        auxifr = auxifr + sys.float_info.epsilon
        auxi[i+1] = auxifr
        print('success to read audio file')
    y_sum = np.zeros(maxlen,)
    print('let add all interference speaker together; maxlen is '+str(maxlen))
    for i in range(len(nbspk)):
      
        if len(y[i])==maxlen:
            y_sum = y_sum+y[i]
        else:
            #print(len(y[i]))
            start = randint(0,maxlen-len(y[i])-1)
            #print(start)
            begpart = np.zeros(start,)
            endpart = np.zeros(maxlen-start-len(y[i]),)
            y_sum = y_sum + np.concatenate((begpart,y[i],endpart), axis=0)
    print('adding is completed and reading_interferingspk is done')
    return y_sum, auxi, my_list
def wiener(mat_s,mat_n):
    line, column = np.shape(mat_s)
    mask = np.zeros((line, column))
    #mask = np.sqrt(mat_s/(mat_s+mat_n))
    mask = np.sqrt(mat_s**2/(mat_s**2+mat_n**2))
    print('mask computation is done')
    return mask
def addup(s,n):
    maxlen = np.maximum(len(s),len(n))
    print('maxlen  and length of signals are '+str(maxlen)+' '+str(len(s))+' '+str(len(n)))
    tmp_n = n
    tmp_s = s
    # append zers if neccesary
    if len(s)==maxlen:
        while len(n)<maxlen:
            n = np.concatenate((n,tmp_n),axis=0)
    else:
        while len(s)<maxlen:
            s = np.concatenate((s,tmp_s),axis=0)
    s = s[0:maxlen]
    n = n[0:maxlen]
    snr = 10*np.log10(np.sum(s**2)/np.sum(n**2))
    db = 0.0
    alpha = np.sqrt(np.sum(s**2)/(np.sum(n**2)*10.0**(db/10.0)))
    # check SNR
    snr1 = 10*np.log10(np.sum(s**2)/np.sum((alpha*n)*(alpha*n)))
    y = s + alpha*n
    print('snr before '+str(snr)+' after '+str(snr1))
    return y, s, alpha*n
def gen_feature(chunklist,malelist,femalelist,featdir):
    # read chunklist
    chunkfilelist = readlineoffile(chunklist)
    # check if the target speaker is male or female
    for i in range(len(chunkfilelist)):
        my_list=list()
        array = chunkfilelist[i].split('/')
        file_name = array[-1][0:-4]
        tgtspk = array[-2]
        gender = array[-3]
        print('reading wav file...')
        print('target speaker is '+tgtspk+' and gender is '+gender+ ' filename is '+ file_name)
        try:
            s, fs = readwavfile(chunkfilelist[i])
        except:
            continue
        print('reading audio at '+chunkfilelist[i])
        print('let pick up two auxiliary utterances from this speaker')
        s = s + sys.float_info.epsilon
        auxlist = readlineoffile('config/spk/'+tgtspk+'.txt')
        randnumber = randint(0,len(auxlist)-1)
        auxfile = auxlist[randnumber]
        try:
            aux1, fs = readwavfile(auxfile)
        except:
            randnumber = randint(0,len(auxlist)-1)
            auxfile = auxlist[randnumber]
            aux1, fs = readwavfile(auxfile)
        print('reading auxiliary audio at '+auxfile)
        my_list.append(auxfile)
        aux1 = aux1 + sys.float_info.epsilon

        # add interspeaker [1] means that one interference speaker with difference gender
        inference_spk_array = [1]
        ifrspk, auxi, my_list = reading_interferingspk(tgtspk,inference_spk_array,gender,my_list)
        auxi[0] = aux1
        
        print('addup...')
        mix, s_fill, ifrspk_fill= addup(s,ifrspk)
	
        # then now compute feature
        print('feature extraction for mixture')
        featmix = featureextract(mix,'main')

        # compute the mask
        print('compute gammatone for target speaker')
        sgm, _ = wav2spec(s_fill)
        print('compute gammatone for other')
        bias = np.random.normal(0, 1, len(ifrspk_fill))
        ngm, _ = wav2spec(ifrspk_fill+0.04*bias)
        print('compute the mask')
        mask = wiener(sgm,ngm)
        print('save file...')
        matdir = featdir+'/'+file_name+'.'+tgtspk+'.'+gender+'.mat'
        txtdir = featdir+'/'+file_name+'.'+tgtspk+'.'+gender+'.txt'
        print('matfile is saved at '+matdir)
        savemat(matdir,{'featmix':featmix,'mask':mask})
        list2file(txtdir,my_list)
        # try to recover signal
        #mixed, est_phase = wav2spec(mix)
        #enh_wav = spec2wav(mask*mixed,est_phase)
        #savemat('mask.mat',{'mask':mask})
        #wavfile.write('idea.wav',16000,enh_wav)
        #exit()

        
        del my_list
    return 1
def main():
    parser = argparse.ArgumentParser(description='Generating feature')
    parser.add_argument('--chunklist', type=str, default='none', metavar='N',
                        help='chunklist file')
    parser.add_argument('--malelist', type=str, default='male.list', metavar='LR',
                        help='male.list')
    parser.add_argument('--femalelist', type=str, default='female.list', metavar='LR',
                        help='female.list')
    parser.add_argument('--nbinfer', type=int, default=1, metavar='LR',
                        help='number of interference')
    args = parser.parse_args()
    featdir = args.chunklist.split('/')[1]
    print(featdir)
    if not os.path.isdir(featdir):
        os.mkdir(featdir)
            
    gen_feature(args.chunklist,args.malelist,args.femalelist,featdir)
    return 1

if __name__ == '__main__':
    main()
