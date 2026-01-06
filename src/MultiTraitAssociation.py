#!/usr/bin/env python3
# Import required modules
import os
import subprocess
import sys
import threading
from typing import List, Union
from utils.CLIFramework import CLIFramework, OptionConfig
from utils.LoggingUtils import log


def stream_subprocess(proc: subprocess.Popen) -> int:
    """Stream stdout/stderr from subprocess to our logger/stdout and wait."""
    stdout_lines = []
    stderr_lines = []

    def _reader(pipe, collect, is_err: bool = False):
        try:
            for ln in iter(pipe.readline, ""):
                if not ln:
                    break
                collect.append(ln)
                sys.stdout.write(ln)
                sys.stdout.flush()
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    t_out = threading.Thread(
        target=_reader, args=(proc.stdout, stdout_lines, False), daemon=True
    )
    t_err = threading.Thread(
        target=_reader, args=(proc.stderr, stderr_lines, True), daemon=True
    )
    t_out.start()
    t_err.start()
    proc.wait()
    t_out.join(timeout=1.0)
    t_err.join(timeout=1.0)
    return proc.returncode


def run_cpassoc(
    input_files: Union[List[str], str],
    correlation_matrix: str,
    sample_sizes: Union[List[int], str],
    output_file: str,
    alpha: float,
    marker_col: str,
    t_col: str,
    cols_to_add: str,
    traits: str,
    log_level: str = "INFO",
) -> str:
    """Run CPASSOC R script for multi-trait association analysis."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rscript_path = os.path.join(script_dir, "utils", "CPASSOC.R")
    rscript_path = os.path.normpath(rscript_path)
    if not os.path.exists(rscript_path):
        raise FileNotFoundError(
            f"CPASSOC R script not found at expected locations: {rscript_path}"
        )

    cmd = [
        "Rscript",
        rscript_path,
        "--input",
        input_files,
        "--correlation_matrix",
        os.path.abspath(correlation_matrix),
        "--sample_sizes",
        sample_sizes,
        "--output_file",
        os.path.abspath(output_file),
        "--marker_col",
        marker_col,
        "--t_col",
        t_col,
        "--cols_to_add",
        cols_to_add,
        "--traits",
        traits,
        "--alpha",
        str(float(alpha)),
        "--log_level",
        log_level,
    ]

    log.debug("Invoking CPASSOC R script")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1,
    )

    rc = stream_subprocess(proc)
    if rc != 0:
        raise RuntimeError(f"CPASSOC R script failed with exit code {rc}")

    log.info(f"CPASSOC finished successfully, results written to {output_file}")
    log.success("Multi-trait association analysis completed.")
    return output_file


options = [
    OptionConfig(flags=["-i", "--input"], type=str, required=True),
    OptionConfig(flags=["-c", "--correlation_matrix"], type=str, required=True),
    OptionConfig(flags=["-s", "--sample_sizes"], type=str, required=True),
    OptionConfig(flags=["-o", "--output_file"], type=str, required=True),
    OptionConfig(flags=["-m", "--marker_col"], type=str, required=True),
    OptionConfig(flags=["-t", "--t_col"], type=str, required=True),
    OptionConfig(flags=["-d", "--cols_to_add"], type=str, required=False),
    OptionConfig(flags=["-r", "--traits"], type=str, default="", required=False),
    OptionConfig(flags=["-a", "--alpha"], type=float, default=0.05, required=False),
]

if __name__ == "__main__":
    framework = CLIFramework(option_list=options, script_name="MultiTraitAssociation")
    opt = framework.run()
    run_cpassoc(
        input_files=opt.input,
        correlation_matrix=opt.correlation_matrix,
        sample_sizes=opt.sample_sizes,
        output_file=opt.output_file,
        alpha=opt.alpha,
        marker_col=opt.marker_col,
        t_col=opt.t_col,
        cols_to_add=opt.cols_to_add,
        traits=opt.traits,
        log_level=opt.verbose,
    )
