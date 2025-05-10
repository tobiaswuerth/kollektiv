import logging
import os
from datetime import datetime
from kollektiv import System


def setup_logging():
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"kollektiv_{timestamp}.log")

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # File handler - logs everything to file
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s\t | %(name)s: %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    # Console handler - logs INFO and above to console
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    logging.info(f"Logging system initialized. Log file: {log_filename}")
    return log_filename


if __name__ == "__main__":
    log_file = setup_logging()
    print(f"Writing log to: {log_file}")
    logging.info("Kollektiv started.")

    goal = (
        "The goal is to write a story. "
        "The story must have 5 chapters and each chapter must consist of around 500 words (±20 words). "
        "The story must be a a novel short story with a plot playing in a post-apocalyptic world. "
        "The required output is a markdown file 'book.md'. "
    )

    # goal = (
    #     "Develop an online Pong game with two-player support over the internet. "
    #     "Include basic mechanics: paddles, bouncing ball, and scoring. "
    #     "Provide client and server files."
    # )

    # from kollektiv.llm.tools import PDFHandler
    # goal = PDFHandler._read_pdf("2025 d_SSA Modul BMDT.pdf")

    logging.info(f"Starting system with goal: {goal[:50]}...")
    System(goal).run()
    logging.info("Kollektiv ended.")
