import logging

class ConditionalFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.formatters = {
            logging.DEBUG: logging.Formatter('DEBUG: %(message)s'),
            logging.INFO: logging.Formatter('%(message)s'),
            logging.WARNING: logging.Formatter('WARNING: %(message)s'),
            logging.ERROR: logging.Formatter('ERROR: %(message)s'),
            logging.CRITICAL: logging.Formatter('CRITICAL: %(message)s')
        }

    def format(self, record):
        formatter = self.formatters.get(record.levelno, self.formatters[logging.DEBUG])
        return formatter.format(record)

class SingletonLogger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SingletonLogger, cls).__new__(cls)
            cls._instance.log = logging.getLogger(__name__)
            cls._instance.log.setLevel(logging.INFO)
            
            formatter = ConditionalFormatter()
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            cls._instance.log.addHandler(handler)
        return cls._instance

    def set_level(self, level):
        self.log.setLevel(level)

logger = SingletonLogger()