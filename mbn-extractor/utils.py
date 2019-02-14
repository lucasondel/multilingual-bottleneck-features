#!/usr/bin/env python

########################################################################################
#  copyright (C) 2017 by Anna Silnova, Pavel Matejka, Oldrich Plchot, Frantisek Grezl  #
#                         Brno Universioty of Technology                               #
#                         Faculty of information technology                            #
#                         Department of Computer Graphics and Multimedia               #
#  email             : {isilnova,matejkap,iplchot,grezl}@vut.cz                        #
########################################################################################
#                                                                                      #
#  This software and provided models can be used freely for research                   #
#  and educational purposes. For any other use, please contact BUT                     #
#  and / or LDC representatives.                                                       #
#                                                                                      #
########################################################################################

import os, errno
import numpy as np
import scipy.fftpack
import scipy.linalg as spl
import numexpr as ne
import logging
import struct
import warnings
warnings.simplefilter("error",RuntimeWarning)

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise
def mkdir_subdirs(seglist, out_dir='./'):
    """
    Create sub-directories relative to the out_dir according to seglist
    """
    for dir in set(map(os.path.dirname, seglist)):
        mkdir_p(out_dir+'/'+dir)

def framing(a, window, shift=1):
    shape = ((a.shape[0] - window) // shift + 1, window) + a.shape[1:]
    strides = (a.strides[0]*shift, a.strides[0]) + a.strides[1:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def dct_basis(nbasis, length):
    # the same DCT as in matlab
    return scipy.fftpack.idct(np.eye(nbasis, length), norm='ortho')

def add_dither(x, level=8):
    np.random.seed(42)
    return x + level * (np.random.rand(*x.shape)*2-1)

def mel_inv(x):
    return (np.exp(x/1127.)-1.)*700.

def mel(x):
    return 1127.*np.log(1. + x/700.)

def mel_fbank_mx(winlen_nfft, fs, NUMCHANS=20, LOFREQ=0.0, HIFREQ=None):
    """Returns mel filterbank as an array (NFFT/2+1 x NUMCHANS)
    winlen_nfft - Typically the window length as used in mfcc_htk() call. It is
    used to determine number of samples for FFT computation (NFFT).
    If positive, the value (window lenght) is rounded up to the
    next higher power of two to obtain HTK-compatible NFFT.
    If negative, NFFT is set to -winlen_nfft. In such case, the
    parameter nfft in mfcc_htk() call should be set likewise.
    fs          - sampling frequency (Hz, i.e. 1e7/SOURCERATE)
    NUMCHANS    - number of filter bank bands
    LOFREQ      - frequency (Hz) where the first filter strats
    HIFREQ      - frequency (Hz) where the last  filter ends (default fs/2)
     """
    if not HIFREQ: HIFREQ = 0.5 * fs
    nfft = 2**int(np.ceil(np.log2(winlen_nfft))) if winlen_nfft > 0 else -int(winlen_nfft)
    fbin_mel = mel(np.arange(nfft / 2 + 1, dtype=float) * fs / nfft)
    cbin_mel = np.linspace(mel(LOFREQ), mel(HIFREQ), NUMCHANS + 2)
    cind = np.floor(mel_inv(cbin_mel) / fs * nfft).astype(int) + 1
    mfb = np.zeros((len(fbin_mel), NUMCHANS))
    for i in range(NUMCHANS):
         mfb[cind[i]  :cind[i+1], i] = (cbin_mel[i]  -fbin_mel[cind[i]  :cind[i+1]]) / (cbin_mel[i]  -cbin_mel[i+1])
         mfb[cind[i+1]:cind[i+2], i] = (cbin_mel[i+2]-fbin_mel[cind[i+1]:cind[i+2]]) / (cbin_mel[i+2]-cbin_mel[i+1])
         if LOFREQ > 0.0 and float(LOFREQ)/fs*nfft+0.5 > cind[0]: mfb[cind[0],:] = 0.0 # Just to be HTK compatible
    return mfb


def fbank_htk(x, window, noverlap, fbank_mx):
    """Mel log Mel-filter bank channel outputs
    Returns NUMCHANS-by-M matrix of log Mel-filter bank outputs extracted from
    signal x, where M is the number of extracted frames, which can be computed
    as floor((length(x)-noverlap)/(window-noverlap)). Remaining parameters
    have the following meaning:
    x         - input signal
    window    - frame window lentgth (in samples, i.e. WINDOWSIZE/SOURCERATE)
    or vector of widow weights
    noverlap  - overlapping between frames (in samples, i.e window-TARGETRATE/SOURCERATE)
    fbank_mx  - array with (Mel) filter bank (as returned by function mel_fbank_mx()).
 """

    if np.isscalar(window):
        window = np.hamming(window)
    nfft = 2**int(np.ceil(np.log2(window.size)))
    x = framing(x.astype("float"), window.size, window.size-noverlap).copy()
    x *= window
    x = np.fft.rfft(x, nfft)
    x = x.real**2 + x.imag**2
    x = np.log(np.maximum(1.0, np.dot(x, fbank_mx)))
    return x


def write_htk(file, m ):
    """ Write htk feature file
    Input:
    file-  file to save features to
    m  - data: one row per frame
    """
    m = np.atleast_2d(m)
    #print m
    try:
        fh = open(file,'wb')
    except TypeError:
        fh = file
    #print fh
    try:
        fh.write(struct.pack(">IIHH", len(m), 0.01*1e7, m.shape[1] *4,9))
        #print  len(m), 0.01*1e7, m.shape[1] * 4,9
        m = m.astype('>f')
        fh.write(m.tobytes())
    finally:
        if fh is not file: fh.close()

def read_htk(file):
    """ Read htk feature file
     Input:
         file: file name or file-like object.
     Outputs:
          m  - data: one row per frame
    """
    try:
        fh = open(file,'rb')
    except TypeError:
        fh = file
    try:
        nSamples, sampPeriod, sampSize, parmKind = struct.unpack(">IIHH", fh.read(12))
        m = np.frombuffer(fh.read(nSamples*sampSize), 'i1')
        m = m.view('>f').reshape(nSamples,sampSize/4)
    finally:
        if fh is not file: fh.close()
    return m

def read_lab_to_bool_vec(lab_file, true_label=None, length=0, frame_rate=100.):
    """
    Read HTK label file into boolean vector representing frame labels
    Inputs:
        lab_file: name of a HTK label file (possibly gzipped)
        true_label: label for which the output frames should have True value (defaul: all labels)
        length: Output vector is truncted or augmented with False values to have this length.
        For negative 'length', it will be only augmented if shorter than '-length'.
        By default (length=0), the vector entds with the last true value.
        frame_rate: frame rate of the output vector (in frames per second)
    Output:
        frames: boolean vector
    """
    min_len, max_len = (length, length) if length > 0 else (-length, None)
    labels = np.atleast_2d(np.loadtxt(lab_file, usecols=(0,1,2), dtype=object))
    if true_label: labels = labels[labels[:,2] == true_label]
    start, end = np.rint(frame_rate/1e7*labels.T[:2].astype(int)).astype(int)
    if not end.size: return np.zeros(min_len, dtype=bool)
    frms = np.repeat(np.r_[np.tile([False,True], len(end)), False], np.r_[np.c_[start - np.r_[0, end[:-1]], end-start].flat, max(0, min_len-end[-1])])
    assert len(frms) >= min_len and np.sum(end-start) == np.sum(frms)
    return frms[:max_len]

def compute_vad(s, win_length=200, win_overlap=120, n_realignment=5, threshold=0.3):
    import gmm
    # power signal for energy computation
    s = s**2
    # frame signal with overlap
    F = framing(s, win_length, win_length - win_overlap)
    # sum frames to get energy
    E = F.sum(axis=1).astype(np.float64)
    # E = np.sqrt(E)
    # E = np.log(E)

    # normalize the energy
    E -= E.mean()
    try:
        E /= E.std()
    # initialization
        mm = np.array((-1.00, 0.00, 1.00))[:, np.newaxis]
        ee = np.array(( 1.00, 1.00, 1.00))[:, np.newaxis]
        ww = np.array(( 0.33, 0.33, 0.33))

        GMM = gmm.gmm_eval_prep(ww, mm, ee)

        E = E[:,np.newaxis]

        for i in range(n_realignment):
        # collect GMM statistics
            llh, N, F, S = gmm.gmm_eval(E, GMM, return_accums=2)

        # update model
            ww, mm, ee   = gmm.gmm_update(N, F, S)
        # wrap model
            GMM = gmm.gmm_eval_prep(ww, mm, ee)

    # evaluate the gmm llhs
        llhs = gmm.gmm_llhs(E, GMM)

        llh  = gmm.logsumexp(llhs, axis=1)[:,np.newaxis]

        llhs = np.exp(llhs - llh)

        out  = np.zeros(llhs.shape[0], dtype=np.bool)
        out[llhs[:,0] < threshold] = True
    except RuntimeWarning:
        logging.info("File contains only silence")
        out=np.zeros(E.shape[0],dtype=np.bool)

    return out
