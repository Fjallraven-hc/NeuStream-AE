"""Logging configuration for vLLM."""
import datetime
import json
import logging
import os
import sys
from functools import partial
from logging import Logger
from logging.config import dictConfig
from os import path
from typing import Dict, Optional, List

import vllm.envs as envs

VLLM_CONFIGURE_LOGGING = envs.VLLM_CONFIGURE_LOGGING
VLLM_LOGGING_CONFIG_PATH = envs.VLLM_LOGGING_CONFIG_PATH
VLLM_LOGGING_LEVEL = envs.VLLM_LOGGING_LEVEL

_FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"

DEFAULT_LOGGING_CONFIG = {
    "formatters": {
        "vllm": {
            "class": "vllm.logging.NewLineFormatter",
            "datefmt": _DATE_FORMAT,
            "format": _FORMAT,
        },
    },
    "handlers": {
        "vllm": {
            "class": "logging.StreamHandler",
            "formatter": "vllm",
            "level": VLLM_LOGGING_LEVEL,
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "vllm": {
            "handlers": ["vllm"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
    "version": 1,
}


def _configure_vllm_root_logger() -> None:
    logging_config: Optional[Dict] = None

    if not VLLM_CONFIGURE_LOGGING and VLLM_LOGGING_CONFIG_PATH:
        raise RuntimeError(
            "VLLM_CONFIGURE_LOGGING evaluated to false, but "
            "VLLM_LOGGING_CONFIG_PATH was given. VLLM_LOGGING_CONFIG_PATH "
            "implies VLLM_CONFIGURE_LOGGING. Please enable "
            "VLLM_CONFIGURE_LOGGING or unset VLLM_LOGGING_CONFIG_PATH.")

    if VLLM_CONFIGURE_LOGGING:
        logging_config = DEFAULT_LOGGING_CONFIG

    if VLLM_LOGGING_CONFIG_PATH:
        if not path.exists(VLLM_LOGGING_CONFIG_PATH):
            raise RuntimeError(
                "Could not load logging config. File does not exist: %s",
                VLLM_LOGGING_CONFIG_PATH)
        with open(VLLM_LOGGING_CONFIG_PATH, encoding="utf-8",
                  mode="r") as file:
            custom_config = json.loads(file.read())

        if not isinstance(custom_config, dict):
            raise ValueError("Invalid logging config. Expected Dict, got %s.",
                             type(custom_config).__name__)
        logging_config = custom_config

    if logging_config:
        dictConfig(logging_config)


def init_logger(name: str) -> Logger:
    """The main purpose of this function is to ensure that loggers are
    retrieved in such a way that we can be sure the root vllm logger has
    already been configured."""

    return logging.getLogger(name)


# The root logger is initialized when the module is imported.
# This is thread-safe as the module is only imported once,
# guaranteed by the Python GIL.
_configure_vllm_root_logger()

logger = init_logger(__name__)


def _trace_calls(log_path, root_dir, frame, event, arg=None):
    if event in ['call', 'return']:
        # Extract the filename, line number, function name, and the code object
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        func_name = frame.f_code.co_name
        if not filename.startswith(root_dir):
            # only log the functions in the vllm root_dir
            return
        # Log every function call or return
        try:
            last_frame = frame.f_back
            if last_frame is not None:
                last_filename = last_frame.f_code.co_filename
                last_lineno = last_frame.f_lineno
                last_func_name = last_frame.f_code.co_name
            else:
                # initial frame
                last_filename = ""
                last_lineno = 0
                last_func_name = ""
            with open(log_path, 'a') as f:
                if event == 'call':
                    f.write(f"{datetime.datetime.now()} Call to"
                            f" {func_name} in {filename}:{lineno}"
                            f" from {last_func_name} in {last_filename}:"
                            f"{last_lineno}\n")
                else:
                    f.write(f"{datetime.datetime.now()} Return from"
                            f" {func_name} in {filename}:{lineno}"
                            f" to {last_func_name} in {last_filename}:"
                            f"{last_lineno}\n")
        except NameError:
            # modules are deleted during shutdown
            pass
    return partial(_trace_calls, log_path, root_dir)


def enable_trace_function_call(log_file_path: str,
                               root_dir: Optional[str] = None):
    """
    Enable tracing of every function call in code under `root_dir`.
    This is useful for debugging hangs or crashes.
    `log_file_path` is the path to the log file.
    `root_dir` is the root directory of the code to trace. If None, it is the
    vllm root directory.

    Note that this call is thread-level, any threads calling this function
    will have the trace enabled. Other threads will not be affected.
    """
    logger.warning(
        "VLLM_TRACE_FUNCTION is enabled. It will record every"
        " function executed by Python. This will slow down the code. It "
        "is suggested to be used for debugging hang or crashes only.")
    logger.info("Trace frame log is saved to %s", log_file_path)
    if root_dir is None:
        # by default, this is the vllm root directory
        root_dir = os.path.dirname(os.path.dirname(__file__))
    sys.settrace(partial(_trace_calls, log_file_path, root_dir))

LOG_DIR = os.environ.get("NEU_LOG_DIR", None)
if LOG_DIR is not None:
    os.makedirs(LOG_DIR, exist_ok=True)
    logger.info(f"Logging to {LOG_DIR}")
else:
    logger.warning(f"ONLY Logging to stdout")

def add_filehandler(logger: logging.Logger, file: str, level=logging.DEBUG):
    if LOG_DIR is None:
        return
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, file))
    file_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    

def add_consolehandler(logger: logging.Logger, level=logging.INFO):
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def setlevel(logger: logging.Logger, level = logging.DEBUG):
    logger.setLevel(level)

from vllm.outputs import RequestOutput
from vllm.core.predictor import Predictor

def output_slo_log(output: RequestOutput, predictor: Predictor, tokenizer) -> str:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    attained, slo, eptime, edtime, info = predictor.check_seq_slo(output)
    res = ">>>"*20+"\n"
    res +=f"prompt len: {len(tokenizer(prompt)['input_ids'])}, generated len: {len(tokenizer(generated_text)['input_ids'])}\n"
    res += f"SLO attained: {attained}.\n\
            arr: {output.arrival_time}\n\
            slo: {slo}(+{slo-output.arrival_time})\n\
            fin: {output.finished_time}(+{output.finished_time-output.arrival_time})\n\
            pslo: {output.pslo}\n\
            dslo: {output.dslo}\n\
            exp ptime: {eptime}\n\
            exp dtime: {edtime}\n"
    res += "<<<"*20
    return res


def print_table(*columns):
    if not columns:
        return

    # 检查所有列长度是否相等
    column_length = len(columns[0])
    if any(len(column) != column_length for column in columns):
        raise ValueError("所有列的长度必须相等")

    # 计算每一列的最大宽度
    column_widths = [max(len(str(item)) for item in column) for column in columns]

    # 打印表格
    table = ""
    for i in range(column_length):
        row = [str(column[i]).ljust(column_widths[idx]) for idx, column in enumerate(columns)]
        table += ('| ' + ' | '.join(row) + ' |' + '\n')
        print('| ' + ' | '.join(row) + ' |')
        
    return table


def slo_chart(outputs: List[RequestOutput], predictor: Predictor, tokenizer, num_reqs):
    names = ["req id", "SLO Attained?", "pmp len", "gen len", "arrival time", "SLO", "finished time", "p slo", "d slo", "excepted ptime", "excepted dtime"]
    ids = ["req id"]
    attained_ls = ["SLO Attained"]
    pmplens = ["pmp len"]
    genlens = ["gen len"]
    arr_times = ["arrival time"]
    slos = ["SLO"]
    fin_time = ["finished time"]
    pslos = ["p slo"]
    dslos = ["d slo"]
    eptimes = ["excepted ptime"]
    edtimes = ["excepted dtime"]


    for output in outputs:
        ids.append(output.request_id)
        prompt = output.prompt
        generated_text = output.outputs[0].text
        attained, slo, eptime, edtime, info = predictor.check_seq_slo(output)
        attained_ls.append(attained)
        arr_times.append(output.arrival_time)
        slos.append("+"+str(slo-output.arrival_time))
        fin_time.append("+"+str(output.finished_time-output.arrival_time))
        eptimes.append(eptime)
        edtimes.append(edtime)
        pmplens.append(len(tokenizer(prompt)['input_ids']))
        # genlens.append(len(tokenizer(generated_text)['input_ids']))
        genlens.append(len(output.outputs[0].token_ids))
        pslos.append(output.pslo)
        dslos.append(output.dslo)

    table = print_table(ids, attained_ls, pmplens, genlens, arr_times, slos, fin_time, pslos, dslos, eptimes, edtimes)
    sloinfo = f"SLO Attained ratio: {sum(attained_ls[1:])}/{len(outputs)} = {sum(attained_ls[1:])/len(outputs)*100:.2f}%"
    print(sloinfo)
    return table, sloinfo, len(outputs)
