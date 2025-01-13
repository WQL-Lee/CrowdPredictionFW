import logging
import logging.config
from pathlib import Path
import os
from utils import read_json


def setup_logging(debug, save_dir='', log_config='logger/logger_config.json', default_level=logging.INFO):
    """
    Setup logging configuration
    """
    if debug:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        logger = logging.getLogger()
        logger.addHandler(console_handler)
    else: 
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        log_config = Path(log_config)
        if log_config.is_file():
            config = read_json(log_config)
            # modify logging paths based on run config
            for _, handler in config['handlers'].items():
                if 'filename' in handler:
                    handler['filename'] = os.path.join(save_dir, handler['filename'])

            logging.config.dictConfig(config)
        else:
            print("Warning: logging configuration file is not found in {}.".format(log_config))
            logging.basicConfig(level=default_level)
