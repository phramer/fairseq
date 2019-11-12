#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
from comet_ml import ExistingExperiment, Experiment

import os
import socket
import subprocess
from getpass import getpass

import keyring

from train import main as single_process_main
from fairseq import distributed_utils, options


def main(args, config=None):
    if args.distributed_init_method is None and args.distributed_port > 0:
        # We can determine the init method automatically for Slurm.
        node_list = os.environ.get("SLURM_JOB_NODELIST")
        if node_list is not None:
            try:
                hostnames = subprocess.check_output(
                    ["scontrol", "show", "hostnames", node_list]
                )
                args.distributed_init_method = "tcp://{host}:{port}".format(
                    host=hostnames.split()[0].decode("utf-8"),
                    port=args.distributed_port,
                )
                args.distributed_rank = int(os.environ.get("SLURM_PROCID"))
                args.device_id = int(os.environ.get("SLURM_LOCALID"))
            except subprocess.CalledProcessError as e:  # scontrol failed
                raise e
            except FileNotFoundError as e:  # Slurm is not installed
                pass
    if args.distributed_init_method is None and args.distributed_port is None:
        raise ValueError(
            "--distributed-init-method or --distributed-port "
            "must be specified for distributed training"
        )

    args.distributed_rank = distributed_utils.distributed_init(args)
    print(
        "| initialized host {} as rank {}".format(
            socket.gethostname(), args.distributed_rank
        )
    )
    single_process_main(args, config=config)


if __name__ == "__main__":
    parser = options.get_training_parser()
    parser.add_argument(
        "--comet-logging",
        action="store_true",
        help="Whether to use Comet.ML for logging",
    )
    args = options.parse_args_and_arch(parser)

    logging = getattr(args, "comet_logging", False)
    config = None
    if logging:
        PROJECT = "phramer"
        if not keyring.get_password("comet", PROJECT):
            comet_ml_api_key = getpass("Please enter the comet.ml API key: ")
            keyring.set_password("comet", PROJECT, comet_ml_api_key)
        else:
            comet_ml_api_key = keyring.get_password("comet", PROJECT)

        experiment = Experiment(
            api_key=comet_ml_api_key,
            project_name="phramer",
            workspace="phramer",
            auto_output_logging=None,
        )
        config = {"api_key": comet_ml_api_key, "experiment_key": experiment.get_key()}
        print("Proceeding with Comet.ML logging...")
    main(args, config=config)
