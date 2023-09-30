import logging


logger = logging.getLogger(__name__)

def log_title_with_multiple_lines(title: str):
    logger.info("")
    logger.info("=" * 80)
    logger.info(" " + title)
    logger.info("=" * 80)
    logger.info("")
