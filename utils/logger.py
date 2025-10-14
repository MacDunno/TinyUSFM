import logging  
import os  
from termcolor import colored 
import logging


def setup_logger(log_dir, log_file_name="train.log"):  
    os.makedirs(log_dir, exist_ok=True)  
    log_file_path = os.path.join(log_dir, log_file_name)  
    logger = logging.getLogger()  
    logger.setLevel(logging.INFO)  
    fmt = "[%(asctime)s %(name)s] (%(filename)s:%(lineno)d): %(levelname)s %(message)s"  
    color_fmt = (  
        colored("[%(asctime)s %(name)s]", "green")  
        + colored(" (%(filename)s:%(lineno)d)", "yellow")  
        + ": %(levelname)s %(message)s"  
    )  
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')  
    file_handler.setLevel(logging.INFO)  
    file_formatter = logging.Formatter(fmt)  
    file_handler.setFormatter(file_formatter)  
    console_handler = logging.StreamHandler()  
    console_handler.setLevel(logging.INFO)  
    class ColoredFormatter(logging.Formatter):  
        def format(self, record):  
            log_msg = super().format(record)  
            return color_fmt % record.__dict__  

    console_formatter = ColoredFormatter(fmt)  
    console_handler.setFormatter(console_formatter)  

    logger.addHandler(file_handler)  
    logger.addHandler(console_handler)  

    return logger