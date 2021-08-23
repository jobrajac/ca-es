Code used for training models and conducting experiments for bachelor thesis.

Overview of files:

- main.py

Main entrypoint to train a model. Run `python main.py --help` for a overview of possible arguments.

- ca_es.py

The class where the code to run an evolution is located. Keeps track of hyperparameters and methods to run and evaluate the model in the environment.

- net.py

The neural network used to train the models.

- dist.py

Used by master and workers to communicate with Redis when running an evolution on multiple machines on the same network. Host and password configuration needs to be changed in the file to point to a running Redis-instance.

- pool.py

Class used to store a set amount of tensors that can be sampled and overwritten.

- utils.py

Utility functions for various small tasks.

- weight_updates.py

Method to change network parameters according to Hebbian rules and post- and presynaptic values.

- test_models.py

File to load saved models and compare them in various experiments. Comment in/out the models in the bottom of the file and change the dmg_func parameter to the desired damage function. The models used in the report are located in models/final

- net_modeltest.py

This file is very similar to net.py, with minor changes to return more information for experiments.


- test_es_consistency.py

Compares the loss for an evolution of models at 40 and 200 update steps. 

- convert_tf_to_pt.py

Convert Tensorflow models to PyTorch models.

