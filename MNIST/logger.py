import logging

def setup_logging(log_file='log.txt'):
    # 获取 root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 清除所有已存在的 handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 文件 Handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)

    # 控制台 Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    # 添加 Handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# 使用示例
if __name__ == "__main__":
    setup_logging()
    logging.debug("This will go to log.txt")
    logging.info("This will go to console and log.txt")
    logging.error("Error message for test")
