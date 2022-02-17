from argparse import ArgumentParser
import numpy as np

def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", '-d', type=str, default='mnist',
        help="""Dataset to run on. Needs to be either a name specified in data_dict in data.py,
                or a path to a dataset that can be read by get_precomputed in data.py"""
    )
    parser.add_argument(
        "--model", '-m', type=str, default='CNN',
        help="Name of the model: see models/__init__.py for names of available models"
    )
    parser.add_argument(
        "--model-parameters", nargs="*", type=int,
        help="The number of neurons/filters to use in the model layers"
    )
    parser.add_argument(
        "--localization", '-l', type=str, default='false',
        help="Name of localization: see models/__init__.py for names of available localizations"
    )
    parser.add_argument(
        "--localization-parameters", nargs="*", type=int,
        help="The number of neurons/filters to use in the localization layers"
    )
    parser.add_argument(
        "--stn-placement", '-p', nargs="*", type=int, default=[],
        help="Number of layers to place stn after"
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Number of time to run this experiment, default 1"
    )
    parser.add_argument(
        "--name", '-n', type=str, default='result',
        help="Name of directory to create and save results in. Must not already exist."
    )
    parser.add_argument(
        "--optimizer", '-o', type=str, default='sgd',
        help="Name of the optimizer: sgd, nesterov, or adam."
    )
    parser.add_argument(
        "--lr", type=float, default=0.01,
        help="Initial learning rate to use. Default is 0.01"
    )
    parser.add_argument(
        "--switch-after-iterations", nargs="*", type=int, default=[np.inf],
        help="""The learning rate is divided by divide-lr-by at each iteration in this list.
                The list must be increasing."""
    )
    parser.add_argument(
        "--loc-lr-multiplier", type=float, default=1,
        help="""The localization network's learning rate is loc-lr-multiplier multiplied
                by the base learning rate."""
    )
    parser.add_argument(
        "--pre-stn-multiplier", type=float, default=1,
        help="""The learning rate of layers that come before at least one localization network is
                pre-stn-multiplier multiplied by the base learning rate."""
    )
    parser.add_argument(
        "--divide-lr-by", type=float, default=10,
        help="Divide the lr by this number after switch_after_iterations."
    )
    parser.add_argument(
        "--momentum", type=float, default="0",
        help="How large the momentum is. (Use optimizer 'nesterov' for nesterov momentum)"
    )
    parser.add_argument(
        "--weight-decay", "-w", type=float, default=0,
    )
    parser.add_argument(
        "--batch-size", '-b', type=int, default=256,
    )
    parser.add_argument(
        "--load-model", type=str, default="",
        help="Initialize the model to that of the path"
    )
    parser.add_argument(
        "--add-iteration", nargs="*", type=int, default = [],
        help="""Add extra st-iterations (i.e. multiple transformations) after processing
                this many batches."""
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed to use. If None, the computation may not be deterministic."
    )

    epoch_parser = parser.add_mutually_exclusive_group(required=True)
    epoch_parser.add_argument(
        "--epochs", '-e', type=int, dest="epochs",
        help="Epochs to train on."
    )
    epoch_parser.add_argument(
        "--iterations", '-i', type=int, dest="iterations",
        help="Number of iterations (i.e. batches) to train on."
    )

    # STN-SLX
    loop_parser = parser.add_mutually_exclusive_group(required=False)
    loop_parser.add_argument(
        "--loop", dest="loop", action="store_true",
        help="Use the localization network's parameters to transform the *input*."
    )
    loop_parser.add_argument(
        '--no-loop', dest='loop', action='store_false',
        help="Use the localization network's parameters to transform the featuremap it's placed after."
    )
    parser.set_defaults(loop=False)

    # STN-DLX
    deep_parser = parser.add_mutually_exclusive_group(required=False)
    deep_parser.add_argument(
        "--deep", dest="deep", action="store_true",
        help="""Include copies of n=stn_placement number of layers from the classification
                network in the localization networks."""
    )
    deep_parser.add_argument(
        '--no-deep', dest='deep', action='store_false',
    )
    parser.set_defaults(deep=False)

    rotate_parser = parser.add_mutually_exclusive_group(required=False)
    rotate_parser.add_argument(
        "--rotate", dest="rotate", action="store_true",
        help="Rotate the data randomly before feeding it to the network"
    )
    rotate_parser.add_argument(
        '--no-rotate', dest='rotate', action='store_false',
    )
    parser.set_defaults(rotate=False)

    normalize_parser = parser.add_mutually_exclusive_group(required=False)
    normalize_parser.add_argument(
        "--normalize", dest="normalize", action="store_true",
        help="Normalize the data to mean 0 std 1."
    )
    normalize_parser.add_argument(
        '--no-normalize', dest='normalize', action='store_false',
    )
    parser.set_defaults(normalize=False)

    batchnorm_parser = parser.add_mutually_exclusive_group(required=False)
    batchnorm_parser.add_argument(
        "--batchnorm", dest="batchnorm", action="store_true",
        help="Use batchnorm after all STNs."
    )
    batchnorm_parser.add_argument(
        '--no-batchnorm', dest='batchnorm', action='store_false',
    )
    parser.set_defaults(batchnorm=False)

    iterative_parser = parser.add_mutually_exclusive_group(required=False)
    iterative_parser.add_argument(
        "--iterative", dest="iterative", action="store_true",
        help="""If a number reoccurs in stn-placement, it will reuse the
                localization parameters iff iterative is True. Default: True"""
    )
    iterative_parser.add_argument(
        '--no-iterative', dest='iterative', action='store_false',
    )
    parser.set_defaults(iterative=True)

    moment_sched_parser = parser.add_mutually_exclusive_group(required=False)
    moment_sched_parser.add_argument(
        "--moment-sched", dest="moment_sched", action="store_true",
        help="""Use image moments to display images of a particular orientation
                more frequently, to enforce a canonical orientation.
                Default: False"""
    )
    moment_sched_parser.add_argument(
        "--no-moment-sched", dest="moment_sched", action="store_false",
    )
    parser.set_defaults(moment_sched=False)

    pretrain_parser = parser.add_mutually_exclusive_group(required=False)
    pretrain_parser.add_argument(
        "--pretrain", dest="pretrain", action="store_true",
        help="""Train the stn on moments instead of training the classifier.
                Default: False"""
    )
    pretrain_parser.add_argument(
        '--no-pretrain', dest='pretrain', action='store_false',
    )
    parser.set_defaults(pretrain=False)

    hook_llr_parser = parser.add_mutually_exclusive_group(required=False)
    hook_llr_parser.add_argument(
        "--hook-llr", dest="hook_llr", action="store_true",
        help="""Use a hook to implement localization learning rate. If True,
                the gradient that goes through an ST will be multiplied by
                localization_learning_rate (and the localization network's
                learning rate will be equal to the base learning rate).
                This means that the layers before the localization network
                will be less affected by the gradient that has passed through
                the ST.
                Default: False"""
    )
    hook_llr_parser.add_argument(
        '--no-hook-llr', dest='hook_llr', action='store_false',
    )
    parser.set_defaults(hook_llr=False)

    onlyloc_parser = parser.add_mutually_exclusive_group(required=False)
    onlyloc_parser.add_argument(
        "--onlyloc", dest="onlyloc", action="store_true",
        help="""If true, do not train the final layers or output
                Default: False"""
    )
    onlyloc_parser.add_argument(
        '--no-onlyloc', dest='onlyloc', action='store_false',
    )
    parser.set_defaults(onlyloc=False)

    return parser
