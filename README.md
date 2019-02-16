# BUT/PHONEXIA BOTTLENECK FEATURE EXTRACTOR

The BUT/PHONEXIA Bottleneck feature extractor is a tool for extracting
bottleneck features or phoneme classes posteriors from audio signal.
The goal is to provide a tool for extracting such features to people who
do not have access to the databases, do not have capacity to train it or
to those who want to use it just as complementary features.

## License

Scripts and model files for the Babel BNF extractor (3) can be used for research and educational purposes only. Scripts and model files for the Fisher-based BNF extractors (1,2) can be used only by participants of the NIST LRE2017 and also only for research and educational purposes. It is explicitly forbidden to use the Fisher-based models by any party which does not have a proper license for the FISHER corpora (LDC2004S13, LDC2004T19, LDC2005S13, LDC2005T19). Any use of the software and models described above must be of non-commercial character. For any other use, please contact BUT and/or LDC representative.

Please see the `LICENSE` file for detailed information


## Models Description
There are 3 pretrained networks provided with this release:

### 1. FisherEnglish_FBANK_HL500_SBN80_PhnStates120

Trained on Fisher English with 120 phoneme states as output classes (40 phonemes, 3 state for each phoneme)
The training corpora are:
  * LDC2004S13 Fisher English Training Speech Part 1 Speech
  * LDC2004T19 Fisher English Training Speech Part 1 Transcripts
  * LDC2005S13 Fisher English Training Part 2, Speech
  * LDC2005T19 Fisher English Training Part 2, Transcripts
   
location: `nn_weights/FisherEnglish_FBANK_HL500_SBN80_PhnStates120.npz`

### 2. FisherEnglish_FBANK_HL500_SBN80_triphones2423
Trained on Fisher English with 2423 triphones as output classes
The training corpora are the same as in 1)

location: `nn_weights/FisherEnglish_FBANK_HL500_SBN80_triphones2423.npz`

### 3. Babel-ML17_FBANK_HL1500_SBN80_PhnStates3096
Trained on 17 languages from [IARPA BABEL project](https://www.iarpa.gov/index.php/research-programs/babel)
The BN is trained as Multilingual Bottleneck with 3096 output classes
(3 phoneme states per each language stacked together). The training corpora are:
  *  LDC2016S02 IARPA Babel Cantonese Language Pack IARPA-babel101b-v0.4c"
  *  LDC2016S06 IARPA Babel Assamese Language Pack IARPA-babel102b-v0.5a" 
  * LDC2016S08 IARPA Babel Bengali Language Pack IARPA-babel103b-v0.4b
  * LDC2016S09 IARPA Babel Pashto Language Pack IARPA-babel104b-v0.4bY
  * LDC2016S10 IARPA Babel Turkish Language Pack IARPA-babel105b-v0.5
  * LDC2016S13 IARPA Babel Tagalog Language Pack IARPA-babel106-v0.2g
  * LDC2017S01 IARPA Babel Vietnamese Language Pack IARPA-babel107b-v0.7
  * LDC2017S03 IARPA Babel Haitian Creole Language Pack IARPA-babel201b-v0.2b
  * LDC2017S08 IARPA Babel Lao Language Pack IARPA-babel203b-v3.1a
  * LDC2017S13 IARPA Babel Tamil Language Pack IARPA-babel204b-v1.1b
  * LDC2017S19 IARPA Babel Zulu Language Pack IARPA-babel206b-v0.1e

And other corpora not yet released by LDC:
  * IARPA Babel Kurmanji (Kurdish) Language Pack IARPA-babel205
  * IARPA Babel TokPisin Language Pack IARPA-babel207
  * IARPA Babel Cebuano Language Pack IARPA-babel301
  * IARPA Babel Kazakh Language Pack IARPA-babel302
  * IARPA Babel Telugu Language Pack IARPA-babel303
  * IARPA Babel Lithuanian Language Pack IARPA-babel304
    
location: `nn_weights/Babel-ML17_FBANK_HL1500_SBN80_PhnStates3096.npz`

All networks were trained using a context of MEL filterbanks input
features extracted with [HTK](http://htk.eng.cam.ac.uk/)
with the following "HCopy" configuration (see
[HTK documentation](http://www.ee.columbia.edu/~dpwe/LabROSA/doc/HTKBook21/node78.html)):
```
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
```

## Requirements

The implementation runs on python 3.7+ and most likely (though not tested) python 3+. Your distribution will need the following packages:
  - numpy
  - scipy
  - numexpr

Note: these dependencies are specify in the `setup.py` file so they should be automatically installed.

## Installation 

To install the script to extract features run in the root directory of the repository:
```
$ python setup.py install
```
## Extracting features

Once the package is installed you can extract the multi-lingual bottleneck features with the utility `audio2bottleneck`. Given a "scp" file, say `example.scp`, formatted as in [Kaldi](https://github.com/kaldi-asr/kaldi) recipes:
```
faem0_si1392 sph2pipe -f wav /mnt/matylda2/data/TIMIT/timit/train/dr2/faem0/si1392.wav | sox - -t wav - rate 8000 |
faem0_si2022 sph2pipe -f wav /mnt/matylda2/data/TIMIT/timit/train/dr2/faem0/si2022.wav | sox - -t wav - rate 8000 |
faem0_si762 sph2pipe -f wav /mnt/matylda2/data/TIMIT/timit/train/dr2/faem0/si762.wav | sox - -t wav - rate 8000 |
faem0_sx132 sph2pipe -f wav /mnt/matylda2/data/TIMIT/timit/train/dr2/faem0/sx132.wav | sox - -t wav - rate 8000 |
faem0_sx222 sph2pipe -f wav /mnt/matylda2/data/TIMIT/timit/train/dr2/faem0/sx222.wav | sox - -t wav - rate 8000 |
faem0_sx312 sph2pipe -f wav /mnt/matylda2/data/TIMIT/timit/train/dr2/faem0/sx312.wav | sox - -t wav - rate 8000 |
faem0_sx402 sph2pipe -f wav /mnt/matylda2/data/TIMIT/timit/train/dr2/faem0/sx402.wav | sox - -t wav - rate 8000 |
faem0_sx42 sph2pipe -f wav /mnt/matylda2/data/TIMIT/timit/train/dr2/faem0/sx42.wav | sox - -t wav - rate 8000 |
fajw0_si1263 sph2pipe -f wav /mnt/matylda2/data/TIMIT/timit/train/dr2/fajw0/si1263.wav | sox - -t wav - rate 8000 |
fajw0_si1893 sph2pipe -f wav /mnt/matylda2/data/TIMIT/timit/train/dr2/fajw0/si1893.wav | sox - -t wav - rate 8000 |
```

you can extract the features for all these file by running:
```
$ mkdir mbnfea
$ audio2bottleneck --model babel-17 --outdir mbnfea/ example.scp	 
```
Or using the pipe command: 
```
$ mkdir mbnfea
$ cat example.scp | audio2bottleneck --model babel-17 --outdir mbnfea/ -
```
Note:
  - if no model is specified, the "babel-17" model will be selected by default
  - the input WAV file has to be sampled at 8 kHz (use `sox - -t wav - rate 8000` to resample your audio files)
  - the `sph2pipe` command in the example above is specific to the data (in this case this is an excerpt of our TIMIT "scp" file) and you will probably not need it if you work with other data set

## Referencing

When using this tool, please kindly cite our analysis paper:

Radek Fer, Pavel Matejka, Frantisek Grezl, Oldrich Plchot, Karel Vesely, Jan Honza Cernocky, Multilingually trained bottleneck features in spoken language recognition, In Computer Speech & Language, Volume 46, Pages 252-267, 2017.

```
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
```

## Authors
* Lucas Ondel iondel@fit.vutbr.cz   (only packaging of the python 3+ scripts)
* Anna Silnova isilnova@fit.vutbr.cz
* Pavel Matejka matejkap@fit.vutbr.cz
* Oldrich Plchot iplchot@fit.vutbr.cz
* Frantisek Grezlgrezl@fit.vutbr.cz  
* Jan "Honza" Cernocky cernocky@fit.vutbr.cz
