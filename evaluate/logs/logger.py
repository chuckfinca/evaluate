import logging
import time

class SingletonLogger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SingletonLogger, cls).__new__(cls)
            cls._instance.log = logging.getLogger(__name__)
            cls._instance.log.setLevel(logging.INFO)
            
            formatter = logging.Formatter('%(levelname)s: %(message)s')
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            cls._instance.log.addHandler(handler)
        return cls._instance

    def set_level(self, level):
        self.log.setLevel(level)

logger = SingletonLogger()