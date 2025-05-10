import pymupdf
import os
import logging

from kollektiv.core import ToolMessage
from kollektiv.config import config


class PDFHandler:
    directory: str = config.output_dir
    logger = logging.getLogger(__name__)

    @staticmethod
    def _read_pdf(file_name: str) -> str:
        PDFHandler.logger.debug(f"Reading PDF file: '{file_name}'...")
        path = os.path.join(PDFHandler.directory, file_name)
        assert os.path.exists(path), f"File '{file_name}' not found."

        with pymupdf.open(path) as doc:
            content = ""
            for page in doc:
                content += page.get_text()

        # clean up the content
        content = " ".join(content.split())
        PDFHandler.logger.debug(f"PDF content extracted: {len(content)} characters")

        return content

    @staticmethod
    def read_pdf(file_name: str) -> ToolMessage:
        """
        Reads the content of a PDF and returns a ToolMessage object with the file's content.
        Args:
            file_name (str): The name of the file to be read.
        Returns:
            ToolMessage: A message indicating the success or failure of the file read operation.
                         If successful, the message contains the file's content.
                         If the file is not found, an error message is returned.
        """
        PDFHandler.logger.info(f"Attempting to read PDF file: '{file_name}'...")
        try:
            content = PDFHandler._read_pdf(file_name)
            PDFHandler.logger.info(
                f"Successfully read PDF '{file_name}' ({len(content)} characters)"
            )
            return ToolMessage(
                (
                    f"Successfully read PDF '{file_name}'.\n"
                    f"<content>\n{content}\n</content>"
                )
            )
        except Exception as e:
            PDFHandler.logger.error(
                f"Error reading PDF '{file_name}': {str(e)}", exc_info=True
            )
            return ToolMessage(
                f"!! [ERROR]: Reading PDF '{file_name}' failed: {str(e)}"
            )
