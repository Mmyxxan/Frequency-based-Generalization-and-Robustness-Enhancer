from datetime import datetime
from .tools import mkdir_if_missing

import logging
logger = logging.getLogger(__name__)

def set_up_logger(cfg):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mkdir_if_missing(f"{cfg.MODEL.OUTPUT_DIR}/log")
    logfile = f"{cfg.MODEL.OUTPUT_DIR}/log/log_{timestamp}.log"

    logging.basicConfig(
        filename=logfile,
        encoding='utf-8',
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

logger.debug('Logger is set up successfully!')