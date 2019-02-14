'Multilingual Bottleneck Feature Extractor.'

import argparse
import io
import logging
import os
import subprocess
import sys

import numpy as np
from scipy.io.wavfile import read

import utils
import nn_def

logging.basicConfig(format= '%(levelname)s: %(message)s',level=logging.INFO)
logger = logging.getLogger()


models = {
    'fisher-monophone': './nn_weights/FisherEnglish_FBANK_HL500_SBN80_PhnStates120.npz',
    'fisher-triphone': './nn_weights/FisherEnglish_FBANK_HL500_SBN80_triphones2423.npz',
    'babel-17l': './nn_weights/Babel-ML17_FBANK_HL1500_SBN80_PhnStates3096.npz'

}

def read_signal(file_name):
    if os.path.isfile(file_name):
        extension= file_name.split('.')[-1]
        if extension=='wav':
            fs,signal=wav.read(file_name)
            if not fs==8000:
                logging.info("Unsupported audio format, expected audio input should be 8kHz")
        elif extension=='raw':
            signal=np.fromfile(file_name,dtype='int16')
        else:
            logging.info('Unvalid file extension, cannot load signal %s',file_name )
            signal=[]
    else:
        logging.info('File %s is missing',file_name)
        signal=[]
    return signal


# Parameters to extract fbank features.
window = np.hamming(200)
fbank_mx = utils.mel_fbank_mx(window.size, fs=8000, NUMCHANS=24,
                              LOFREQ=64.0, HIFREQ=3800.0)

def extract_fbank(signal, noverlap=120):
    signal = utils.add_dither(signal, 0.1)
    fea = utils.fbank_htk(signal, window, noverlap, fbank_mx)
    return fea


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m', '--model', default='fisher-monophone',
                        choices=[name for name in models],
                        help='pretrained model to extract the features')
    parser.add_argument('-o', '--outdir', default=os.getcwd(),
                        help='output directory (default: ./)')
    parser.add_argument('--vad', help='user defined VAD (deactivate the '\
                                      'internal VAD')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='verbose output')
    parser.add_argument('wavlist', help='list of WAV files or "-" for stdin')
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.debug(f'model: {args.model}')
    logger.debug(f'outdir: {args.outdir}')
    logger.debug(f'vad: {args.vad}')

    logger.debug('load the model')
    nn_weights = np.load(models[args.model])
    global_context = right_ctx = 15
    context_1st_bn = nn_weights['context']

    if args.wavlist == '-':
        infile = sys.stdin
    else:
        with open(args.wavlist, 'r') as f:
            infile = f.readlines()

    counts = 0
    for line in infile:
        tokens = line.strip().split()
        uttid, inwav = tokens[0], ' '.join(tokens[1:])

        # If 'inwav' ends up with the '|' symbol, 'inwav' is
        # interpreted as a command otherwise we assume 'inwav' to
        # be a path to a wav file.
        if inwav[-1] == '|':
            cmd = inwav[:-1]
            logger.debug(f'reading command: {cmd}')
            proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
            sr, signal = read(io.BytesIO(proc.stdout))
        else:
            logger.debug(f'reading file: {inwav}')
            sr, signal = read(inwav)

        # The models were trained on 8kHz data, we do not process audio
        # file with other sampling rate.
        if not sr == 8000:
            logger.error(f'unsupported sampling rate {sr}, skipping {inwav}')
            continue

        # Extract FBANK features.
        fea = extract_fbank(signal)

        # Voice Activity Detection
        if args.vad:
            logger.debug(f'using user vad: {args.vad}')
            vad = utils.read_lab_to_bool_vec(vad_file,length=len(fea))
        else:
            logging.debug('using energy-based VAD')
            vad = utils.compute_vad(signal)
            logging.debug("%d frames of speech detected", sum(vad))

        if sum(vad)==0:
            logging.warning('no speech detected, no features will be created')
            continue

        logger.debug('FBANK features mean normalization')
        fea -= np.mean(fea[vad],axis=0)

        logger.debug(f'stacking features context={global_context}')
        fea = np.r_[np.repeat(fea[[0]], global_context, axis=0), fea,
                    np.repeat(fea[[-1]], global_context, axis=0)]

        logger.debug(f'pre-processing features nnet input')
        nn_input = nn_def.preprocess_nn_input(fea, context_1st_bn,
                                              context_1st_bn)

        # Extract the MBN features.
        _, bn_fea = nn_def.create_nn_extract_st_BN(nn_input, nn_weights)
        pad = global_context - context_1st_bn
        bn_fea = np.vstack(bn_fea)[pad:-pad]

        path = os.path.join(args.outdir, f'{uttid}.npy')
        logger.debug(f'saving the features to {path}')
        np.save(path, bn_fea)

        counts += 1

    logger.info(f'extracted features for {counts} utterances')

if __name__ == '__main__':
    run()
else:
    logger.error('This script cannot be imported !')
    exit(1)

