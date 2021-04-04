"""Build logger"""
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)
