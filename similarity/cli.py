from .utils.config import Config
from .experiment import Experiment
from pathlib import Path
import numpy as np
import logging
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .utils.config import BaseConfig
    from argparse import ArgumentParser, Namespace


def get_argparser(config_cls: type["BaseConfig"]) -> "ArgumentParser":
    parser = config_cls.argparser()
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "-l", "--log-file", nargs="?", type=Path, help="Path to log file"
    )
    return parser


def setup_logging(args: "Namespace") -> logging.Logger:
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler.setFormatter(formatter)
    logger = logging.getLogger("similarity")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger = logging.getLogger(__name__)
    logger.debug("Parsed arguments: %s", args)
    return logger


def experiment() -> None:
    p = get_argparser(Config)
    p.add_argument(
        "-o", "--output-file", nargs="?", type=Path, help="Path to output TSV file"
    )
    p.add_argument(
        "-p",
        "--peptide-file",
        nargs="?",
        type=Path,
        help="Path to output peptide table",
    )
    p.add_argument(
        "-a",
        "--array-file",
        nargs="?",
        type=Path,
        help="Path to output .npy file with raw score arrays",
    )
    args = p.parse_args()
    logger = setup_logging(args)
    kw = vars(args).copy()
    for key in ["verbose", "output_file", "peptide_file", "array_file", "log_file"]:
        kw.pop(key)

    config = Config(**kw)
    with Experiment(config) as exp:
        if args.output_file:
            exp.score_df.to_csv(args.output_file, index=False, sep="\t")
            logger.info("Saved results to %s", args.output_file)

        if args.peptide_file:
            exp.peptides.to_csv(args.peptide_file, index=False, sep="\t")
            logger.info("Saved peptide table to %s", args.peptide_file)

        if args.array_file:
            np.save(args.array_file, exp.score_array)
            logger.info("Saved raw score arrays to %s", args.array_file)


def time_scoring() -> None:
    p = get_argparser(Config)

    p.add_argument(
        "-a",
        "--array-file",
        nargs="?",
        type=Path,
        help="Path to output .npy file with raw score arrays",
    )
    args = p.parse_args()

    logger = setup_logging(args)
    kw = vars(args).copy()
    for key in ["verbose", "array_file", "log_file"]:
        kw.pop(key, None)

    config = Config(**kw)
    with Experiment(config) as exp:
        _ = exp.predicted_spectra  # Ensure spectra are predicted before timing
        logger.info("Timing the scoring...")
        start_time = datetime.now()
        arr = exp.score_array
        elapsed = datetime.now() - start_time
        logger.info("Scoring completed in %s", elapsed)
        if args.array_file:
            np.save(args.array_file, arr)
            logger.info("Saved raw score arrays to %s", args.array_file)


if __name__ == "__main__":
    experiment()
