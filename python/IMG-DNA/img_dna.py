#!/usr/bin/python3

import sys
import os

import getopt

import logger
import img_load

import sift
import orb

model_type = 'sift'
args = None


def _usage_dl_dna():
    print("""\
Usage: img_dna.py [<options>]
                             <image name> : show DNA
                             <image name> <image name>: get similarity
                             <image pair file>: get similarities
   <options>
   -h: help(this message)
   -m <model>: sift, orb
   -X: converts the image to grayscale
   -x <resolution>: image is resized to the specified resolution after loading
   -f <image folder> (IMAGE_FOLDER)
""")


def _setup_envs():
    if 'IMAGE_FOLDER' in os.environ:
        img_load.image_fpath = os.environ['IMAGE_FOLDER']


def _parse_args():
    global model_type, args

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hm:Xx:")
    except getopt.GetoptError:
        logger.error("invalid option")
        _usage_dl_dna()
        exit(1)
    for o, a in opts:
        if o == '-h':
            _usage_dl_dna()
            exit(0)
        elif o == '-m':
            model_type = a
        elif o == '-X':
            img_load.grayscaled = True
        elif o == '-x':
            img_load.resized = int(a)


def get_img_dna_model():
    if model_type == 'sift':
        return sift.SIFT()
    elif model_type == 'orb':
        return orb.ORB()
    print(f"invalid model type: {model_type}")
    exit(1)


if __name__ == "__main__":
    logger.init("img_dna")

    _setup_envs()
    _parse_args()
    model = get_img_dna_model()

    if len(args) > 0:
        if os.path.isfile(args[0]):
            model.show_similarities(args[0])
        elif len(args) == 1:
            model.show_dna(args[0])
        else:
            model.show_similarity(args[0], args[1])