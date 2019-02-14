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

import sys, os, logging
import numpy as np
import h5py
import utils
import nn_def
logging.basicConfig( format= '%(message)s',level=logging.INFO)

if len(sys.argv)==1:
    logging.info( "The BUT Bottleneck feature extractor is a tool for extracting phoneme classes posteriors from bottleneck features.")
    logging.info('Usage: python bottleneck2posterior.py nn_weights input.fea output.h5\n ** nn_weights\t weights of the pre-trained Neural Network.\n    Options are:\n\t* FisherMono -should be used if input bottleneck features were extracted with FisherMono option\n\t* FisherTri- should be used if input bottleneck features were extracted with FisherTri option \n\t* BabelMulti - should be used if input bottleneck features were extracted with BabelMulti option\n ** input.fea\tBN feature file extracted by audio2bottleneck.py\n ** output.h5\toutput file, default format for the output is h5 file.\n\tThere is possibility to save the output to HTK feature file (can be done only for FisherMono to size limitations of HTK) or to text file.\n\tIn order to do so, uncomment lines 69 (HTK file) or 70 (text file) of this script.\n')
    sys.exit()
elif len(sys.argv)==4:
    nn,bn_input,post_output=sys.argv[1:4]
else:
    logging.info( "Wrong number of input arguments. 3 are expected")
    sys.exit()

out_dir=os.path.dirname(post_output)
try:
    utils.mkdir_p(out_dir)
except:
    pass
    
#load correct weights
if nn=="FisherMono":
    logging.info('Using NN trained on Fisher English with 120 phone states as targets to calculate posteriors')
    nn='FisherEnglish_SBN80_PhnStates120'
elif nn=="FisherTri":
    logging.info('Using NN trained on Fisher English with 2423 senones as targets to calculate posteriors')
    nn='FisherEnglish_SBN80_triphones2423'
elif nn=="BabelMulti":
    logging.info('Using multilingual NN trained on 17 BABEL languages with language specific phone states as targets to calculate posteriors')
    nn='Babel-ML17_SBN80_PhnStates3096'
else:
    logging.info('Unknown option %s for NN weights, cannot extract posteriors. Valid options are: FisherMono, FisherTri, BabelMulti',nn)
    sys.exit()

if os.path.dirname(sys.argv[0])=='':
    nn_weights='nn_weights/'+nn+'.npz'
else:
    nn_weights=os.path.dirname(sys.argv[0])+'/nn_weights/'+nn+'.npz'
nn_weights=np.load(nn_weights)
#check if the features are already there
if os.path.isfile(post_output):
    logging.info("Posteriors for file %s are already extracted ", bn_input)
else:
    if not os.path.isfile(bn_input): logging.info("Could not read BN feature file %s, cannot extract posteriors", bn_input)
    else:
    	bn=utils.read_htk(bn_input)
    	if nn=="Babel-ML17_SBN80_PhnStates3096":
    	    post=np.vstack(nn_def.create_nn_extract_posterior_ml(bn,nn_weights))
    	else:
    	    post=np.vstack(nn_def.create_nn_extract_posterior(bn,nn_weights))
	with h5py.File(post_output,'w') as f:
	    f.create_dataset('posterior',data=post)
	#utils.write_htk(post_output+'.fea',post) #uncomment to save posteriors in HTK format, can be used only for the FisherEnglish_SBN80_PhnStates120 network because HTK has a limitation on the size of the features
	#np.savetxt(post_output+'.txt', post) #uncomment to save posteriors as a text file
        logging.info('Phoneme classes posteriors are successfully generated for file %s', bn_input )
