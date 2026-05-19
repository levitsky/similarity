from .utils.config import Config, cache_args, CacheConfigType
from .utils.cache import CacheType
from .utils.utils import ExperimentRunner
from .experiment import Experiment
from pathlib import Path
import numpy as np
import logging
from enum import EnumType
from types import UnionType
from datetime import datetime
from dataclasses import fields
from typing import TYPE_CHECKING, Any

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
    return logger


def parse_args(
    parser: "ArgumentParser",
) -> tuple["Namespace", dict[str, Any], logging.Logger]:
    """
    Normalize command-line arguments, sets up logging and returns the namespace and a dict with converted configuration arguments.
    """
    args = parser.parse_args()
    if (
        hasattr(args, "jobs")
        and args.jobs is not None
        and not getattr(args, "all", False)
    ):
        parser.error("--jobs can only be used together with --all")
    logger = setup_logging(args)
    logger.debug("Parsed arguments: %s", args)
    kw = vars(args).copy()
    for key in [
        "verbose",
        "output_file",
        "peptide_file",
        "load_peptide_table",
        "array_file",
        "log_file",
        "all",
        "jobs",
    ]:
        kw.pop(key, None)
    logger.debug("Registered cache configuration arguments: %s", cache_args)

    for field in fields(Config):
        ftype = field.type
        if isinstance(field.type, UnionType):
            ftype = field.type.__args__[0]
        if isinstance(ftype, EnumType):
            value = kw.get(field.name, None)
            if value is not None:
                try:
                    kw[field.name] = ftype[value]
                    logger.debug(
                        "Converted argument %s to enum: %s", field.name, kw[field.name]
                    )
                except KeyError:
                    logger.error(
                        "Invalid value for %s: %s. Expected one of: %s",
                        field.name,
                        value,
                        list(e.name for e in ftype),
                    )
                    raise ValueError(f"Invalid value for {field.name}: {value}")

    cache_type = kw.get("cache", CacheType.NONE)
    if cache_type != CacheType.NONE:
        cache_kw = {}
        for k, v in cache_args.items():
            value = kw.pop(k, None)
            for cct, field in v:
                if cct.name == cache_type.name:
                    cache_kw[k] = (
                        field.type(value) if value is not None else field.default
                    )
                    break
        logger.debug("Cache configuration arguments: %s", cache_kw)
        kw["cache_conf"] = CacheConfigType[cache_type.name].value(**cache_kw)
    else:
        kw["cache_conf"] = None
        for k in cache_args.keys():
            kw.pop(k, None)
    return args, kw, logger


def experiment() -> None:
    p = get_argparser(Config)
    p.add_argument(
        "-o", "--output-file", nargs="?", type=Path, help="Path to output TSV file"
    )
    peptides = p.add_mutually_exclusive_group()
    peptides.add_argument(
        "-p",
        "--peptide-file",
        nargs="?",
        type=Path,
        help="Path to save the peptide table",
    )
    peptides.add_argument(
        "-lp",
        "--load-peptide-table",
        nargs="?",
        type=Path,
        help="Load an existing peptide table",
    )
    p.add_argument(
        "-a",
        "--array-file",
        nargs="?",
        type=Path,
        help="Path to output .npy file with raw score arrays",
    )
    args, kw, logger = parse_args(p)

    config = Config(**kw)
    if args.all:
        if args.peptide_file is None and args.load_peptide_table is None:
            p.error("--all requires either --peptide-file or --load-peptide-table")
        runner = ExperimentRunner(
            config=config,
            peptide_table=args.peptide_file or args.load_peptide_table,
            jobs=args.jobs or 1,
            create_peptide_table=args.peptide_file is not None,
            array_file=str(args.array_file) if args.array_file else None,
            score_df_file=str(args.output_file) if args.output_file else None,
        )
        runner.run()
        return

    with Experiment(config, peptide_table=args.load_peptide_table) as exp:
        if args.peptide_file:
            df = exp.peptides
            df["peptide_sequences"] = df["peptide_sequences"].str.decode("ascii")
            df.to_csv(args.peptide_file, index=False, sep="\t")
            logger.info("Saved peptide table to %s", args.peptide_file)

        if args.output_file:
            exp.score_df.to_csv(args.output_file, index=False, sep="\t")
            logger.info("Saved results to %s", args.output_file)

        if args.array_file:
            np.save(args.array_file, exp.score_array)
            logger.info("Saved raw score arrays to %s", args.array_file)


def time_scoring() -> None:
    p = get_argparser(Config)
    peptides = p.add_mutually_exclusive_group()
    peptides.add_argument(
        "-p",
        "--peptide-file",
        nargs="?",
        type=Path,
        help="Path to save the peptide table",
    )
    peptides.add_argument(
        "-lp",
        "--load-peptide-table",
        nargs="?",
        type=Path,
        help="Load an existing peptide table",
    )
    p.add_argument(
        "-a",
        "--array-file",
        nargs="?",
        type=Path,
        help="Path to output .npy file with raw score arrays",
    )
    args, kw, logger = parse_args(p)

    config = Config(**kw)
    with Experiment(config, peptide_table=args.load_peptide_table) as exp:
        if args.peptide_file:
            df = exp.peptides
            df["peptide_sequences"] = df["peptide_sequences"].str.decode("ascii")
            df.to_csv(args.peptide_file, index=False, sep="\t")
            logger.info("Saved peptide table to %s", args.peptide_file)
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
