import os
import logging

from kollektiv.core import ToolMessage
from kollektiv.config import config


class Storage:
    directory: str = config.output_dir
    logger = logging.getLogger(__name__)

    @staticmethod
    def list_files() -> ToolMessage:
        Storage.logger.debug(f"Listing files in directory: {Storage.directory}")
        result = os.listdir(Storage.directory)
        Storage.logger.debug(f"Files found: {result}")
        return result

    @staticmethod
    def get_files() -> ToolMessage:
        """
        Retrieves a list of files from the specified storage directory.
        Returns:
            ToolMessage: A message containing the list of files if any are found,
            or a message indicating that no files were found.
        """
        Storage.logger.info(f"Getting files from directory: {Storage.directory}...")
        files = Storage.list_files()
        if not files:
            Storage.logger.info("No files found in directory")
            return ToolMessage("No files found.")

        file_list = "\n".join(f"- {file}" for file in files)
        Storage.logger.info(f"Found {len(files)} files")
        Storage.logger.debug(f"Files found: {file_list}")
        return ToolMessage(f"Files found:\n{file_list}")

    @staticmethod
    def read_file(file_name: str) -> ToolMessage:
        """
        Reads the content of a file and returns a ToolMessage object with the file's content.
        Args:
            file_name (str): The name of the file to be read.
        Returns:
            ToolMessage: A message indicating the success or failure of the file read operation.
                         If successful, the message contains the file's content.
                         If the file is not found, an error message is returned.
        """
        Storage.logger.info(f"Attempting to read file: '{file_name}'...")
        file_path = os.path.join(Storage.directory, file_name)
        if not os.path.exists(file_path):
            Storage.logger.warning(f"File not found: '{file_path}'")
            return ToolMessage(f"!! [ERROR]: File '{file_name}' not found.")

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
            
            Storage.logger.info(f"Successfully read file: '{file_name}' ({len(content)} bytes)")
            Storage.logger.debug(f"File content: {content}")
            return ToolMessage(
                (
                    f"Successfully read file '{file_name}'.\n"
                    f"<content>\n{content}\n</content>"
                )
            )
        except Exception as e:
            Storage.logger.error(f"Error reading file '{file_name}': {str(e)}", exc_info=True)
            return ToolMessage(f"!! [ERROR]: Failed to read file '{file_name}': {str(e)}")

    @staticmethod
    def write_file(file_name: str, content: str) -> ToolMessage:
        """
        Writes the given content to a file with the specified name.
        Args:
            file_name (str): The name of the file to be created or overwritten.
            content (str): The content to be written to the file.
        Returns:
            ToolMessage: A message indicating the success or failure of the file write operation.
                         If successful, a success message is returned.
                         If validation fails, an error message is returned.
        """
        Storage.logger.info(f"Attempting to write to file: '{file_name}'...")
        if not file_name:
            Storage.logger.warning("File name cannot be empty.")
            return ToolMessage("!! [ERROR]: File name cannot be empty.")
        if len(file_name) > 50:
            Storage.logger.warning("File name is too long. Maximum 50 characters.")
            return ToolMessage(
                "!! [ERROR]: File name is too long. Maximum 50 characters."
            )
        if not content:
            Storage.logger.warning("Content cannot be empty.")
            return ToolMessage("!! [ERROR]: Content cannot be empty.")

        if not any(
            file_name.endswith(ext) for ext in config.valid_tool_output_extensions
        ):
            Storage.logger.warning(f"Invalid file extension, {file_name}")
            return ToolMessage(
                (
                    "!! [ERROR]: Invalid file extension.\n"
                    f"!! Allowed extensions are: {config.valid_tool_output_extensions}"
                )
            )

        file_path = os.path.join(Storage.directory, file_name)
        os.makedirs(Storage.directory, exist_ok=True)

        try:
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(content)
            
            Storage.logger.info(f"Successfully wrote content to file '{file_name}'")
            Storage.logger.debug(f"Content written: {content}")
            return ToolMessage(f"Successfully wrote content to file '{file_name}'.")
        except Exception as e:
            Storage.logger.error(f"Error writing to file '{file_name}': {str(e)}", exc_info=True)
            return ToolMessage(f"!! [ERROR]: Failed to write to file '{file_name}': {str(e)}")

    @staticmethod
    def count_words(file_name: str) -> ToolMessage:
        """
        Counts the number of words in a file.
        Args:
            file_name (str): The name of the file to be read.
        Returns:
            ToolMessage: A message indicating the success or failure of the word count operation.
                         If successful, the message contains the word count.
                         If the file is not found, an error message is returned.
        """
        Storage.logger.info(f"Counting words in file: '{file_name}'...")
        file_path = os.path.join(Storage.directory, file_name)
        if not os.path.exists(file_path):
            Storage.logger.warning(f"File not found: '{file_path}'")
            return ToolMessage(f"!! [ERROR]: File '{file_name}' not found.")

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                word_count = len(content.split())
            
            Storage.logger.info(f"Word count for '{file_name}': {word_count} words")
            return ToolMessage(f"Word count for '{file_name}': {word_count} words")
        except Exception as e:
            Storage.logger.error(f"Error counting words in file '{file_name}': {str(e)}", exc_info=True)
            return ToolMessage(f"!! [ERROR]: Failed to count words in file '{file_name}': {str(e)}")
