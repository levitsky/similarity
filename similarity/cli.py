from .utils.config import Config, cache_args, CacheConfigType
from .utils.cache import CacheType
from .utils.utils import SingleInputExperimentRunner
from .experiment import SingleInputExperiment, DualInputExperiment
from pathlib import Path
import numpy as np
import logging
from enum import EnumType
from types import UnionType
from datetime import datetime
from dataclasses import fields
from typing import TYPE_CHECKING, Any
from argparse import ArgumentParser

if TYPE_CHECKING:
    from collections.abc import Sequence
    from .utils.config import BaseConfig
    from argparse import Namespace


def get_argparser(
    config_cls: type["BaseConfig"], suffixes: "Sequence[str] | None" = None
) -> "ArgumentParser":
    parser = ArgumentParser()
    files = parser.add_argument_group("Input and output files")
    for suffix in suffixes or [""]:
        suffix = suffix.replace("_", "-")
        files.add_argument(
            f"-i{suffix.lstrip('-')}",
            f"--input-file{suffix}",
            type=Path,
            help=f"Path to input peptide list {suffix.strip('-')}",
        )
        peptides = files.add_mutually_exclusive_group()
        peptides.add_argument(
            f"-p{suffix.lstrip('-')}",
            f"--peptide-file{suffix}",
            type=Path,
            help=f"Path to save the peptide table {suffix.strip('-')}",
        )
        peptides.add_argument(
            f"-lp{suffix.lstrip('-')}",
            f"--load-peptide-table{suffix}",
            type=Path,
            help=f"Load an existing peptide table {suffix.strip('-')}",
        )
        spectra = files.add_mutually_exclusive_group()
        spectra.add_argument(
            f"-s{suffix.lstrip('-')}",
            f"--spectrum-file{suffix}",
            type=Path,
            help=f"Path to save the predicted spectra {suffix.strip('-')} as a .npy file",
        )
        spectra.add_argument(
            f"-ls{suffix.lstrip('-')}",
            f"--load-spectrum-file{suffix}",
            type=Path,
            help=f"Load an existing predicted spectra {suffix.strip('-')} .npy file",
        )
    files.add_argument(
        "-a",
        "--array-file",
        type=Path,
        help="Path to output .npy file with raw score arrays",
    )
    files.add_argument("-o", "--output-file", type=Path, help="Path to output TSV file")
    logsettings = parser.add_argument_group("Logging settings")
    logsettings.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )
    logsettings.add_argument("-l", "--log-file", type=Path, help="Path to log file")
    return config_cls.argparser(parser)


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
    parser: "ArgumentParser", suffixes: "Sequence[str] | None" = None
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
    exclude_keys = ["verbose", "output_file", "array_file", "log_file"]
    for suffix in suffixes or [""]:
        for key in [
            "input_file",
            "peptide_file",
            "spectrum_file",
            "load_peptide_table",
            "load_spectrum_file",
        ]:
            exclude_keys.append(f"{key}{suffix}")
    for key in exclude_keys:
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


def single_input_experiment() -> None:
    p = get_argparser(Config)
    args, kw, logger = parse_args(p)

    config = Config(**kw)
    if args.subsets > 1 and args.subset == 0:
        if args.peptide_file is None and args.load_peptide_table is None:
            p.error(
                "Running multiple subsets requires either --peptide-file or --load-peptide-table"
            )
        runner = SingleInputExperimentRunner(
            config=config,
            input_file=args.input_file,
            peptide_table=args.peptide_file or args.load_peptide_table,
            create_peptide_table=args.peptide_file is not None,
            spectrum_file=args.spectrum_file or args.load_spectrum_file,
            create_spectrum_file=args.spectrum_file is not None,
            array_file=str(args.array_file) if args.array_file else None,
            score_df_file=str(args.output_file) if args.output_file else None,
        )
        runner.run()
        return

    with SingleInputExperiment(
        config,
        input_file=args.input_file,
        peptide_table=args.load_peptide_table,
        spectrum_file=args.load_spectrum_file,
    ) as exp:
        if args.peptide_file:
            df = exp.peptides
            df["peptide_sequences"] = df["peptide_sequences"].str.decode("ascii")
            df.to_csv(args.peptide_file, index=False, sep="\t")
            logger.info("Saved peptide table to %s", args.peptide_file)

        if args.spectrum_file:
            exp.predicted_spectra.save(args.spectrum_file)

        if args.output_file:
            exp.score_df.to_csv(args.output_file, index=False, sep="\t")
            logger.info("Saved results to %s", args.output_file)

        if args.array_file:
            np.save(args.array_file, exp.score_array)
            logger.info("Saved raw score arrays to %s", args.array_file)


def dual_input_experiment() -> None:
    suffixes = ("_1", "_2")
    p = get_argparser(Config, suffixes)
    args, kw, logger = parse_args(p, suffixes)

    config = Config(**kw)

    with DualInputExperiment(
        config,
        input_file_1=args.input_file_1,
        input_file_2=args.input_file_2,
        peptide_table_1=args.load_peptide_table_1,
        peptide_table_2=args.load_peptide_table_2,
        spectrum_file_1=args.load_spectrum_file_1,
        spectrum_file_2=args.load_spectrum_file_2,
    ) as exp:
        for suffix in suffixes:
            if pep_file := getattr(args, f"peptide_file{suffix}"):
                df = getattr(exp, f"peptides{suffix}")
                df["peptide_sequences"] = df["peptide_sequences"].str.decode("ascii")
                df.to_csv(pep_file, index=False, sep="\t")
                logger.info(
                    "Saved peptide table to %s", getattr(args, f"peptide_file{suffix}")
                )

        for suffix in suffixes:
            if spec_file := getattr(args, f"spectrum_file{suffix}"):
                getattr(exp, f"predicted_spectra{suffix}").save(spec_file)

        if args.array_file:
            np.save(args.array_file, exp.score_array)
            logger.info("Saved raw score arrays to %s", args.array_file)

        if args.output_file:
            exp.score_df.to_csv(args.output_file, index=False, sep="\t")
            logger.info("Saved results to %s", args.output_file)


def time_scoring_single() -> None:
    p = get_argparser(Config)
    args, kw, logger = parse_args(p)

    config = Config(**kw)
    with SingleInputExperiment(
        config,
        input_file=args.input_file,
        peptide_table=args.load_peptide_table,
        spectrum_file=args.load_spectrum_file,
    ) as exp:
        if args.peptide_file:
            df = exp.peptides
            df["peptide_sequences"] = df["peptide_sequences"].str.decode("ascii")
            df.to_csv(args.peptide_file, index=False, sep="\t")
            logger.info("Saved peptide table to %s", args.peptide_file)
        if args.spectrum_file:
            exp.predicted_spectra.save(args.spectrum_file)
            logger.info("Saved predicted spectra to %s", args.spectrum_file)
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
    single_input_experiment()
