#!/usr/bin/python3

import sys
import os
import getopt

import logger
import img_load
import dl_dna_model
import triplet_model
import autoencoder_model
import mobilenet

model_type = 'triplet_loss'
fpath_train: str = ""
path_save: str = ""
path_load: str = ""
args = None


def _usage_dl_dna():
    print("""\
Usage: dl-dna.py [<options>]
                             <image name> : show DNA
                             <image name> <image name>: get similarity
                             <image pair file>: get similarities
   <options>
   -h: help(this message)
   -m <model>: triplet_loss(default), mobilenet, autoencoder
   -t <training file>: training mode with file path
       triplet_loss: image list with triple fields
       mobilenet: not supported
   -s <path for save>: path for saving model
   -l <path for load>: path for loading model
   -u <units>: unit count for embedding vector (N_UNITS: env variable)
   -f <image folder> (IMAGE_FOLDER)
   -e <epochs> (EPOCHS)
   -S <seed> (SEED): random seed for deterministic run
   -b <batch_size>: 0 means full batch
   -v <options>: enable verbose output
      k: keras output
      t: triple loss output
      T: simple triple loss output
""")


def _setup_envs():
    if 'N_UNITS' in os.environ:
        dl_dna_model.n_units = int(os.environ['N_UNITS'])
    if 'IMAGE_FOLDER' in os.environ:
        img_load.image_fpath = os.environ['IMAGE_FOLDER']
    if 'EPOCHS' in os.environ:
        dl_dna_model.epochs = int(os.environ['EPOCHS'])
    if 'SEED' in os.environ:
        dl_dna_model.seed = int(os.environ['SEED'])
    if 'FULL_BATCH' in os.environ:
        triplet_model.full_batch = True


def _parse_args():
    global model_type, fpath_train, args, path_save, path_load

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hm:t:f:s:l:u:e:S:b:v:")
    except getopt.GetoptError:
        logger.error("invalid option")
        _usage_dl_dna()
        exit(1)
    for o, a in opts:
        if o == '-h':
            _usage_dl_dna()
            exit(0)
        if o == '-m':
            model_type = a
        if o == '-t':
            fpath_train = a
        elif o == '-f':
            img_load.image_fpath = a
        elif o == '-s':
            path_save = a
        elif o == '-l':
            path_load = a
        elif o == '-u':
            dl_dna_model.n_units = int(a)
        elif o == '-e':
            dl_dna_model.epochs = int(a)
        elif o == '-S':
            dl_dna_model.seed = int(a)
        elif o == '-b':
            dl_dna_model.batch_size = int(a)
        elif o == '-v':
            dl_dna_model.verbose = a


def get_dl_dna_model():
    if model_type == 'triplet_loss':
        return triplet_model.ModelTriplet()
    elif model_type == 'autoencoder':
        return autoencoder_model.AutoEncoder()
    return mobilenet.ModelMobileNet()


if __name__ == "__main__":
    logger.init("dl_dna")

    _setup_envs()
    _parse_args()
    model = get_dl_dna_model()
    if path_load:
        model.load(path_load)
    if fpath_train:
        model.train(fpath_train)
    if path_save:
        model.save(path_save)

    if len(args) > 0:
        if os.path.isfile(args[0]):
            model.show_similarities(args[0])
        elif len(args) == 1:
            model.show_dna(args[0])
        else:
            model.show_similarity(args[0], args[1])
