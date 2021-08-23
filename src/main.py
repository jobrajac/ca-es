import argparse
import os
import shutil
from ca_es import EvolutionStrategy
import time
import multiprocessing as mp
import torch
import json
import sys
import numpy as np
import logging


def main(args, set_by_user):
    if args.cross_machine and args.master:
        args.pop_size = args.pop_size * args.nodes

    if args.pre_trained == "":
        print("Running %s with total pop_size %s" % ("Hebbian" if args.hebb else "ES", args.pop_size))
        print("Cross-machine: ", args.cross_machine)

    # Load configurations for pre-trained model
    if args.pre_trained != "":
        print("Loading pretrained model from %s" % args.pre_trained)
        original_args = argparse.Namespace(**vars(args))
        original_log_folder = args.log_folder
        pre_trained_root = args.pre_trained.split("/")[:3]
        pre_trained_root = "/".join(pre_trained_root)
        with open(pre_trained_root + '/commandline_args.txt', 'r') as f:
            args.__dict__ = json.load(f)
            pre_trained_args = argparse.Namespace(**vars(args))
        args.log_folder = original_args.log_folder
        args.pre_trained = original_args.pre_trained
        args.use_mp = original_args.use_mp
        for elem in set_by_user:
            setattr(args, elem, getattr(original_args, elem))

        if pre_trained_args.log_folder.split("/")[-1] == args.log_folder:
            raise ValueError("You need to specify a log folder that is different from the one containing the pre trained model")
        print("Running %s with total pop_size %s" % ("Hebbian" if args.hebb else "ES", args.pop_size))
    
    # LOGGING
    args.log_folder = "../logs/" + args.log_folder
    LOG_FOLDER = args.log_folder
    if os.path.isdir(LOG_FOLDER):
        shutil.rmtree(LOG_FOLDER)
    os.mkdir(LOG_FOLDER)
    os.mkdir(LOG_FOLDER + "/progress")
    os.mkdir(LOG_FOLDER + "/models")
    os.mkdir(LOG_FOLDER + "/raw")

    # Log command args/hyperparameters
    with open(LOG_FOLDER + '/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    es = EvolutionStrategy(args)

    torch.set_num_threads(1) # Disable parallelization of pytorch calculations
              
    tic = time.time()
    if args.use_mp:
        print("Running with multiprocessing")
        # print("Running with %d threads" % (args.num_threads))¨

    if args.master and not args.cross_machine:
        raise ValueError("Can't run master when cross machine is not enabled.")
    if args.cross_machine and args.master:
        es.run_master()
    else:
        es.run()
    toc = time.time()
    print("Milliseconds used running %d iterations, use_mp %r, pop_size %d: %s" % (args.iter, args.use_mp, args.pop_size, str(int((toc-tic)*1000))))


def str_to_bool(x):
    """Return bool from string"""
    return x == "True"


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Run main")

    # Run options
    parser.add_argument("--log_folder", type=str, default="train_log_testing", metavar="train_log", help="Name of folder to log to")
    parser.add_argument("--hebb", type=str_to_bool, default="False", metavar="False", help="Run with hebb")
    parser.add_argument("--pool", type=str_to_bool, default="False", metavar="False", help="Run with pool")
    parser.add_argument("--use_mp", type=str_to_bool, default="True", metavar="True", help="Run with multiprocessing")
    parser.add_argument("--pre_trained", type=str, default="", metavar="../logs/train_log_testing/models/saved_model_X.pt", help="Path to pretrained model/coeffs")
    
    # Visuals
    parser.add_argument("--emoji", type=str, default="🍉", metavar="🍉", help="Emoji to grow")

    # Image options
    parser.add_argument("--channels", type=int, default=16, metavar=16, help="Amount of channels, including rgba")
    parser.add_argument("--size", type=int, default=9, metavar=9, help="Target size")
    parser.add_argument("--pad", type=int, default=0, metavar=0, help="Padding around target size")
    parser.add_argument("--fire_rate", type=float, default=0.5, metavar=0.5, help="Cell fire rate")
    parser.add_argument("--damage", type=int, default=0, metavar=0, help="Damage n images of the batch every iteration")
    parser.add_argument("--damageChannels", type=str_to_bool, default="False", metavar="False", help="Damage n channels some iterations")

    # Network and general
    parser.add_argument("--lr", type=float, default=0.005, metavar=0.005, help="Learning rate") 
    parser.add_argument("--sigma", type=float, default=0.01, metavar=0.01, help="Sigma")
    parser.add_argument("--iter", type=int, default=1000000, metavar=1000000, help="Total iterations/generations")
    parser.add_argument("--pop_size", type=int, default=6, metavar=6, help="Population size")
    parser.add_argument("--hidden_size", type=int, default=32, metavar=32, help="Amount of neurons in hidden layer")

    # Cross-machine parallelization
    parser.add_argument("--cross_machine", type=str_to_bool, default="False", metavar="False", help="Run while communicating with other machines")
    parser.add_argument("--master", type=str_to_bool, default="False", metavar="False", help="Run as master")
    parser.add_argument("--nodes", type=int, default=1, metavar=5, help="Amount of nodes master node should expect results from")

    args = parser.parse_args()
    set_by_user = []
    for i in range(len(sys.argv)):
        if sys.argv[i][:2] == "--":
            set_by_user.append(sys.argv[i][2:])

    try:
        main(args, set_by_user)
    except Exception as e:
        logging.exception(e)