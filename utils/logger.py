from datetime import datetime
from .ostools import mkdir_if_missing

import logging
logger = logging.getLogger("MyLogger")
# logger.propagate = False

def set_up_logger(cfg):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mkdir_if_missing(f"{cfg.MODEL.OUTPUT_DIR}/log")
    logfile = f"{cfg.MODEL.OUTPUT_DIR}/log/log_{timestamp}.log"

    # logging.basicConfig(
    #     filename=logfile,
    #     encoding='utf-8',
    #     # level=logging.DEBUG,
    #     level=logging.INFO,
    #     # format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    #     format="%(message)s"
    # )

    # --- Remove existing handlers to avoid duplicates ---
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    # --- File handler ---
    file_handler = logging.FileHandler(logfile, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(message)s"))

    # --- Console handler ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    # --- Add handlers ---
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Logger set up successfully!")
