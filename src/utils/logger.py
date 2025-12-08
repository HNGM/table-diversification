import os
import logging
import logging.handlers
from pathlib import Path

log = None
LOG_DIR = Path('logs')

def create_logger(log_folder, log_file_name, timestamp):
    log_file_path = os.path.join(log_folder, log_file_name)

    logger = logging.getLogger(log_file_name)
    logger.setLevel(logging.INFO)  # Set the default level to INFO

    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)  # Set the file handler level to INFO

    formatter = logging.Formatter('%(asctime)s - %(process)d - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Clear existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(file_handler)

    # Suppress logs from secific libraries by setting their level to ERROR
    logging.getLogger('azure.identity').setLevel(logging.ERROR)
    logging.getLogger('azure.identity._internal').setLevel(logging.ERROR)
    logging.getLogger('azure.identity._credentials').setLevel(logging.ERROR)
    logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.ERROR)
    logging.getLogger('httpx').setLevel(logging.ERROR)
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    logging.getLogger('requests.packages.urllib3').setLevel(logging.ERROR)
    logging.getLogger('requests').setLevel(logging.ERROR)

    # Disable all child loggers of urllib3, e.g. urllib3.connectionpool
    logging.getLogger("urllib3").propagate = False

    return logger


def configure_logger(log_file_name, log_timestamp):
    global log

    log_file_name = f'{log_file_name}.log'

    if 'worker' in log_file_name:
        log_folder = LOG_DIR / Path(str(log_timestamp))
        log = create_logger(log_folder, log_file_name, log_timestamp)

    else:
        # Ensure the logs directory exists
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)

        log_folder = LOG_DIR / Path(str(log_timestamp))

        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        log = create_logger(log_folder, log_file_name, log_timestamp)
    return log, log_timestamp
