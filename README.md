# BUT/PHONEXIA BOTTLENECK FEATURE EXTRACTOR

The BUT/PHONEXIA Bottleneck feature extractor is a tool for extracting
bottleneck features or phoneme classes posteriors from audio signal.
The goal is to provide a tool for extracting such features to people who
do not have access to the databases, do not have capacity to train it or
to those who want to use it just as complementary features.

## Authors
* [Lucas Ondel](iondel@fit.vutbr.cz) 
* [Pavel Matejka](matejkap@fit.vutbr.cz)
* [Anna Silnova](isilnova@fit.vutbr.cz)
* [Oldrich Plchot](iplchot@fit.vutbr.cz)
* [Frantisek Grezl](grezl@fit.vutbr.cz)
* [Jan "Honza" Cernocky](cernocky@fit.vutbr.cz)

## License
The models (pretrained networks) are released for noncommercial usage
under CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/)
and python code under Apache 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
More information in LICENSE


## Models Description
All networks with an 80-dimensional bottleneck layer are trained using a
context of MEL filterbanks input features (HCopy configuration file is
below).

There are 3 pretrained networks provided with this release:

### 1. FisherEnglish_FBANK_HL500_SBN80_PhnStates120.npz

Trained on Fisher English with 120 phoneme states as output classes (40 phonemes, 3 state for each phoneme)
The training corpora are:
   * LDC2004S13 Fisher English Training Speech Part 1 Speech
   * LDC2004T19 Fisher English Training Speech Part 1 Transcripts
   * LDC2005S13 Fisher English Training Part 2, Speech
   * LDC2005T19 Fisher English Training Part 2, Transcripts
   
location: `nn_weights/FisherEnglish_FBANK_HL500_SBN80_PhnStates120.npz`

### 2. FisherEnglish_FBANK_HL500_SBN80_triphones2423.npz
Trained on Fisher English with 2423 triphones as output classes
The training corpora are the same as in 1)

location: `nn_weights/FisherEnglish_FBANK_HL500_SBN80_triphones2423.npz`

### 3. Babel-ML17_FBANK_HL1500_SBN80_PhnStates3096.npz
Trained on 17 languages from [IARPA BABEL project](https://www.iarpa.gov/index.php/research-programs/babel)
The BN is trained as Multilingual Bottleneck with 3096 output classes
(3 phoneme states per each language stacked together).
The training corpora are:
    * LDC2016S02 IARPA Babel Cantonese Language Pack IARPA-babel101b-v0.4c
    * LDC2016S06 IARPA Babel Assamese Language Pack IARPA-babel102b-v0.5a
    * LDC2016S08 IARPA Babel Bengali Language Pack IARPA-babel103b-v0.4b
    * LDC2016S09 IARPA Babel Pashto Language Pack IARPA-babel104b-v0.4bY
    * LDC2016S10 IARPA Babel Turkish Language Pack IARPA-babel105b-v0.5
    * LDC2016S13 IARPA Babel Tagalog Language Pack IARPA-babel106-v0.2g
    * LDC2017S01 IARPA Babel Vietnamese Language Pack IARPA-babel107b-v0.7
    * LDC2017S03 IARPA Babel Haitian Creole Language Pack IARPA-babel201b-v0.2b
    * LDC2017S08 IARPA Babel Lao Language Pack IARPA-babel203b-v3.1a
    * LDC2017S13 IARPA Babel Tamil Language Pack IARPA-babel204b-v1.1b
    * LDC2017S19 IARPA Babel Zulu Language Pack IARPA-babel206b-v0.1e

And other corpora not yet released by LDC
    * IARPA Babel Kurmanji (Kurdish) Language Pack IARPA-babel205
    * IARPA Babel TokPisin Language Pack IARPA-babel207
    * IARPA Babel Cebuano Language Pack IARPA-babel301
    * IARPA Babel Kazakh Language Pack IARPA-babel302
    * IARPA Babel Telugu Language Pack IARPA-babel303
    * IARPA Babel Lithuanian Language Pack IARPA-babel304
    
location: `nn_weights/Babel-ML17_FBANK_HL1500_SBN80_PhnStates3096.npz`

## Feature Extraction

In addition, for each bottleneck feature extractor, we provide the
script and models to extract final posterior probabilities.

======================================================
All of the python scripts are developed under python 2.7

USAGE AND FILE DESCRIPTION

audio2bottleneck.py

The script accepts 3 or 4 parameters
python audio2bottleneck.py nn_weights input.wav output.fea vad.lab.gz

nn_weights  ... weights of the pre-trained Neural Network.
                Options are:
                - FisherMono
                - FisherTri
                - BabelMulti
input.wav   ... input audio file in wave format - only sampling frequency 8kHz
                and linear16bit coding accepted
output.fea  ... output feature file in HTK format
vad.lab.gz  ... optional parameter - input label file in HTK label file format
                all labels are considered as speech
                - if this is not provided internal energy based VAD is computed
                - VAD is used for to perform mean normalization of input features (global mean per input segment)


bottleneck2posterior.py

The script accepts 3 parameters
python bottleneck2posterior.py nn_weights input.fea output.h5

nn_weights  ... weights of the pre-trained Neural Network.
		The choice of the network should be consistent with the network used to extract BN features
                Options are:
                - FisherMono (corresponds to FisherMono option of SBN extraction)
                - FisherTri (corresponds to FisherTri)
                - BabelMulti (corresponds to BabelMulti)
input.fea  ... BN feature file - the output of the audio2bottleneck.py
output.h5  ... output file to save posteriors to. Default format is h5 file. The result is saved to the dataset named 'posterior' in the output.h5 file.
	       There is possibility to save the output to HTK feature file (can be done only for FisherEnglish_SBN80_PhnStates120 due to size limitations of HTK) or to text file.
	       In order to do so, uncomment lines 69 (HTK file) or 70 (text file) of the script.


See directory "example" and script run.sh for more information and example how to use it.

Included files:
README - this file
license.txt - information about license
nn_weights/*.npz - files containing neural network weight matrices.
nn_weights/*.dict - files containing target labels used for training.
nn_def.py - functions needed for nn operating
utils.py - other functions needed
gmm.py - GMM functionality, used when vad is computed
audio2bottleneck.py - script for extracting bn features from audio (.raw or .wav) files using defined neural network weights
bottleneck2posterior.py - script for extracting phoneme classes posteriors from bn features extracted with audio2bottleneck.py


======================================================
CITATION

When using this tool, please kindly cite our analysis paper:

Radek Fer, Pavel Matejka, Frantisek Grezl, Oldrich Plchot, Karel Vesely, Jan Honza Cernocky, Multilingually trained bottleneck features in spoken language recognition, In Computer Speech & Language, Volume 46, Pages 252-267, 2017.


@article{Fer:CSL:2017,
title = "Multilingually trained bottleneck features in spoken language recognition",
journal = "Computer Speech & Language",
volume = "46",
number = "Supplement C",
pages = "252 - 267",
year = "2017",
issn = "0885-2308",
author = "Radek Fer and Pavel Matejka and Frantisek Grezl and Oldrich Plchot and Karel Vesely and Jan Honza Cernocky",
}

======================================================
LICENSE

Scripts and model files for the Babel BNF extractor (3) can be used for research and educational purposes only. Scripts and model files for the Fisher-based BNF extractors (1,2) can be used only by participants of the NIST LRE2017 and also only for research and educational purposes. It is explicitly forbidden to use the Fisher-based models by any party which does not have a proper license for the FISHER corpora (LDC2004S13, LDC2004T19, LDC2005S13, LDC2005T19). Any use of the software and models described above must be of non-commercial character. For any other use, please contact BUT and/or LDC representative.


======================================================
HCopy configuration file for the MEL Filter bank extraction

SOURCEKIND   = WAVEFORM
SOURCEFORMAT = WAV
TARGETFORMAT = HTK
TARGETKIND   = FBANK
LOFREQ       = 64
HIFREQ       = 3800
NUMCHANS     = 24       # number of critical bands
USEPOWER     = T        # using power spectrum
USEHAMMING   = T        # use hamming window on speech frame
ENORMALISE   = F
PREEMCOEF    = 0        # no preemphase
TARGETRATE   = 100000   # 10 ms frame rate
WINDOWSIZE   = 250000   # 25 ms window
SAVEWITHCRC  = F
#CEPLIFTER   = 22
NUMCEPS      = 12
WARPFREQ     = 1
WARPLCUTOFF  = 3000
WARPUCUTOFF  = 3000
ADDDITHER    = 0.1












