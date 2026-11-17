import logging
import os

class OptionalFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)
        self.use_format = True  # 默认使用格式

    def format(self, record):
        if self.use_format:
            return super().format(record)
        else:
            return record.getMessage()

class Log(object):
    def __init__(self,head,filename,file='./log/',mode='w'):
        self.logger = logging.getLogger(head)
        self.logger.setLevel(level=logging.INFO)
        if not os.path.exists(file):
            os.makedirs(file)
        handler = logging.FileHandler(file+filename+'.log',mode=mode)
        self.formatter = OptionalFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(self.formatter)
        self.logger.addHandler(handler)
    
    def get(self):
        return self.logger, self.formatter
