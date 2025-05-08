import pymupdf
import os

from ..messages import ToolMessage
from ...config import config


class PDFReader:
    directory: str = config.output_dir

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
        file_path = os.path.join(PDFReader.directory, file_name)
        if not os.path.exists(file_path):
            return ToolMessage(f"!! [ERROR]: File '{file_name}' not found.")

        with pymupdf.open("example.pdf") as doc:
            content = ""
            for page in doc:
                content += page.get_text()

        return ToolMessage(
            (
                f"Successfully read PDF '{file_name}'.\n"
                f"<content>\n{content}\n</content>"
            )
        )
