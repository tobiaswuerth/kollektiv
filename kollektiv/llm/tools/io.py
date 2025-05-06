import os

from ..messages import ToolMessage


class Storage:
    directory: str = "output"

    @staticmethod
    def list_files() -> ToolMessage:
        return os.listdir(Storage.directory)

    @staticmethod
    def get_files() -> ToolMessage:
        """
        Retrieves a list of files from the specified storage directory.
        Returns:
            ToolMessage: A message containing the list of files if any are found,
            or a message indicating that no files were found.
        """
        files = Storage.list_files()
        if not files:
            return ToolMessage("No files found.")

        file_list = "\n".join(f"- {file}" for file in files)
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
        file_path = os.path.join(Storage.directory, file_name)
        if not os.path.exists(file_path):
            return ToolMessage(f"!! [ERROR]: File '{file_name}' not found.")

        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        return ToolMessage(
            (
                f"Successfully read file '{file_name}'.\n"
                f"<content>\n{content}\n</content>"
            )
        )

    @staticmethod
    def write_file(file_name: str, content: str) -> ToolMessage:
        """
        Writes the given content to a file with the specified name.
        Args:
            file_name (str): The name of the file to be created or overwritten (valid extensions are [".txt", ".md", ".json", ".py", ".zip"]).
            content (str): The content to be written to the file.
        Returns:
            ToolMessage: A message indicating the success or failure of the file write operation.
                         If successful, a success message is returned.
                         If validation fails, an error message is returned.
        """
        if not file_name:
            return ToolMessage("!! [ERROR]: File name cannot be empty.")
        if len(file_name) > 50:
            return ToolMessage(
                "!! [ERROR]: File name is too long. Maximum 50 characters."
            )
        if not content:
            return ToolMessage("!! [ERROR]: Content cannot be empty.")

        VALID_EXT = [".txt", ".md", ".json", ".py", ".zip"]
        if not any(file_name.endswith(ext) for ext in VALID_EXT):
            return ToolMessage(
                (
                    "!! [ERROR]: Invalid file extension.\n"
                    f"!! Allowed extensions are: [ {', '.join(VALID_EXT)} ]"
                )
            )

        file_path = os.path.join(Storage.directory, file_name)
        os.makedirs(Storage.directory, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)

        return ToolMessage(f"Successfully wrote content to file '{file_name}'.")

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
        file_path = os.path.join(Storage.directory, file_name)
        if not os.path.exists(file_path):
            return ToolMessage(f"!! [ERROR]: File '{file_name}' not found.")

        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            word_count = len(content.split())

        return ToolMessage(f"Word count for '{file_name}': {word_count} words.")