# Understanding when spatial transformer networks do not support invariance, and what to do about it

This is the code used for the experiments in https://arxiv.org/abs/2004.11678

If you use this code, please cite it as:

L. Finnveden, Y. Jansson and T. Lindeberg (2021) "Understanding when spatial transformer networks do not support invariance, and what to do about it", Proc. International Conference on Pattern Recognition (ICPR 2020), pages 3427-3434, extended version in arXiv:2004.11678.

In order to train models, run `stn/learn.py` in a terminal with appropriate parameters. Descriptions of the parameters can be found by running `python stn/learn.py -h` or by reading `stn/parser.py`. For example, to train a model
* on the MNIST dataset,
* for 10 epochs,
* with a CNN classifier,
* using a spatial transformer network with a CNN localization network,
* which starts at the 0th layer of the classification network; i.e., transforms the image before passing it to the classification network,

you can run `python stn/learn.py -d mnist -e 10 -m CNN -l CNN -p 0`.

The core code, necessary for training models, is in `learn.py`, `parser.py`, `data.py`, `angles.py`, and in the `models` directory. There are various helper functions for evaluating and understanding the trained model in `eval.py`, though these are unfortunately not particularly well-documented. The code in `utils` is not used by any other file, but was written to be run independently, for particular use-cases.

If you get an error in a DataLoader worker process, a good debugging step is to set num_workers to 0 in the relevant dataloader (which will be somewhere in data.py).

Another common error is that learn.py will try to save results to a directory path that already exists. In this case, change the --name parameter (to save to a different directory) or just delete the already-existing directory.

If you have any questions about how to use the code, feel free to ask me on my gmail address, which starts with finnveden.lukas 
