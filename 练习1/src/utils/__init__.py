# -*- coding: utf-8 -*-
from datetime import datetime
from .utils import *
import os
from src.mylib.const import log_dir

log_name = datetime.now().strftime("%m%d%H%M") + ".log"
logger = create_logger(os.path.join(log_dir, log_name))
