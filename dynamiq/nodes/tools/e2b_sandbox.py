import io
from hashlib import sha256
from typing import Any, Literal

from e2b import Sandbox

from dynamiq.connections import E2B as E2BConnection
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents import FileDataModel
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION = """
<tool_description>
This tool enables interaction with an E2B sandbox environment,
providing capabilities for Python code execution and shell command execution.
It offers internet access, API request functionality, and filesystem operations.
Key Features:
1. Python Code Execution
2. Shell Command Execution
3. Internet Access
4. API Request Capability
5. Filesystem Access (Read/Write)
Usage Instructions:
1. Shell Command Execution:
   - Provide the command in the 'shell_command' field.
2. Python Code Execution:
   - Provide the Python code in the 'python' field.
   - Optionally specify packages to install in the 'packages' field (comma-separated).
   - The code will be executed in a clean environment with default packages installed.
   - IMPORTANT: Always write the whole code from beginning to end, including imports.
   - IMPORTANT: Always print the final result when executing Python code.
Notes:
- For API requests, use either shell commands or Python code.
- Filesystem operations can be performed using appropriate Python libraries or shell commands.
Example:
# Python code execution
python = '''
import requests

response = requests.get('https://api.example.com/data')
print(response.json())
'''
# Shell command execution
shell_command = 'ls -la /path/to/directory'
# Package installation (optional)
packages = 'requests,pandas'
</tool_description>
"""  # noqa: E501


def generate_unique_file_name(file: bytes | io.BytesIO, file_description: str = "") -> str:
    """
    Generates a unique file name based on the file description and a hash of the file content.

    Args:
        file: The file content as bytes or BytesIO.
        file_description: The description of the file (used as a prefix).

    Returns:
        A unique file name based on the description and content hash.
    """
    if isinstance(file, io.BytesIO):
        file.seek(0)
        file_content = file.read()
        file.seek(0)
    else:
        file_content = file

    file_hash = sha256(file_content).hexdigest()

    description_part = file_description.replace(" ", "_") if file_description else "file"

    unique_file_name = f"{description_part}_{file_hash[:8]}.bin"
    return unique_file_name


class E2BInterpreterTool(ConnectionNode):
    """
    A tool to interact with an E2B sandbox, allowing for file upload/download,
    Python code execution, and shell command execution.

    Attributes:
        group (Literal[NodeGroup.TOOLS]): Node group type.
        name (str): Name of the tool.
        description (str): Detailed description of the tool's capabilities.
        connection (E2BConnection): E2B connection object.
        installed_packages: list: List of default packages to install.
        files: list[FileDataModel] | None: List of files to upload.
        persistent_sandbox: bool: Whether to use a persistent sandbox across executions.
        _sandbox (Optional[Sandbox]): Persistent sandbox instance (if enabled).
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "code-interpreter_e2b"
    description: str = DESCRIPTION
    connection: E2BConnection
    installed_packages: list = []
    files: list[FileDataModel] | None = None
    persistent_sandbox: bool = True
    _sandbox: Sandbox | None = None
    supports_files: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.persistent_sandbox:
            self._initialize_persistent_sandbox()
        else:
            logger.debug(f"Tool {self.name} - {self.id}: Will initialize sandbox on each execute")

    def _initialize_persistent_sandbox(self):
        """Initializes the persistent sandbox, installs packages, and uploads initial files."""
        logger.info(f"Tool {self.name} - {self.id}: Initializing Persistent Sandbox")
        self._sandbox = Sandbox(api_key=self.connection.api_key)
        self._install_default_packages(self._sandbox)
        if self.files:
            self._upload_files(files=self.files, sandbox=self._sandbox)
            self._update_description()

    def _install_default_packages(self, sandbox: Sandbox) -> None:
        """Installs the default packages in the specified sandbox."""
        if self.installed_packages:
            for package in self.installed_packages:
                self._install_packages(sandbox, package)

    def _install_packages(self, sandbox: Sandbox, packages: str) -> None:
        """Installs the specified packages in the given sandbox."""
        if packages:
            logger.debug(f"Tool {self.name} - {self.id}: Installing packages: {packages}")
            sandbox.process.start_and_wait(f"pip install -qq {' '.join(packages.split(','))}")

    def _upload_files(self, files: list[FileDataModel], sandbox: Sandbox) -> str:
        """Uploads multiple files to the sandbox, generating unique names, and returns details for each file."""
        upload_details = []
        for file_model in files:
            unique_file_name = generate_unique_file_name(file_model.file, file_model.description)
            uploaded_path = self._upload_file(file_model.file, unique_file_name, sandbox)
            upload_details.append(
                {
                    "original_name": unique_file_name,
                    "description": file_model.description,
                    "uploaded_path": uploaded_path,
                }
            )
            logger.debug(f"Tool {self.name} - {self.id}: Uploaded file '{unique_file_name}' to {uploaded_path}")

        self._update_description_with_files(upload_details)

        return "\n".join([f"{file['original_name']} -> {file['uploaded_path']}" for file in upload_details])

    def _upload_file(self, file: bytes | io.BytesIO, file_description: str = "", sandbox: Sandbox | None = None) -> str:
        """Uploads a single file to the specified sandbox and returns the uploaded path."""
        if not sandbox:
            raise ValueError("Sandbox instance is required for file upload.")

        unique_file_name = generate_unique_file_name(file, file_description)

        # Handle the file types (bytes or io.BytesIO)
        if isinstance(file, bytes):
            file_like_object = io.BytesIO(file)
            file_like_object.name = unique_file_name
        elif isinstance(file, io.BytesIO):
            file.name = unique_file_name
            file_like_object = file
        else:
            raise ToolExecutionException(
                f"Error: Invalid file data type: {type(file)}. Expected bytes or BytesIO.", recoverable=False
            )

        # Upload the file to the sandbox
        uploaded_path = sandbox.upload_file(file=file_like_object)
        logger.debug(f"Tool {self.name} - {self.id}: Uploaded file to {uploaded_path}")

        return uploaded_path

    def _update_description_with_files(self, upload_details: list[dict]) -> None:
        """Updates the tool description with detailed information about the uploaded files."""
        if upload_details:
            self.description = self.description.strip().replace("</tool_description>", "")
            self.description += "\n\n**Uploaded Files Details:**"
            for file_info in upload_details:
                self.description += (
                    f"\n- **Original File Name**: {file_info['original_name']}\n"
                    f"  **Description**: {file_info['description']}\n"
                    f"  **Uploaded Path**: {file_info['uploaded_path']}\n"
                )
            self.description += "\n</tool_description>"

    def _execute_python_code(self, code: str, sandbox: Sandbox | None = None) -> str:
        """Executes Python code in the specified sandbox."""
        if not sandbox:
            raise ValueError("Sandbox instance is required for code execution.")
        code_hash = sha256(code.encode()).hexdigest()
        filename = f"/home/user/{code_hash}.py"
        sandbox.filesystem.write(filename, code)
        process = sandbox.process.start_and_wait(f"python3 {filename}")
        if not (process.stdout or process.stderr):
            raise ToolExecutionException(
                "Error: No output. Please use 'print()' to display the result of your Python code.",
                recoverable=True,
            )
        if "Error" in process.stderr:
            raise ToolExecutionException(f"Error during Python code execution: {process.stderr}", recoverable=True)
        return process.stdout

    def _execute_shell_command(self, command: str, sandbox: Sandbox | None = None) -> str:
        """Executes a shell command in the specified sandbox."""
        if not sandbox:
            raise ValueError("Sandbox instance is required for command execution.")

        process = sandbox.process.start(command)
        output = process.wait()
        if process.exit_code != 0:
            raise ToolExecutionException(f"Error during shell command execution: {output.stderr}", recoverable=True)
        return output.stdout

    def execute(self, input_data: dict[str, Any], config: RunnableConfig | None = None, **kwargs) -> dict[str, Any]:
        """Executes the requested action based on the input data."""
        logger.debug(f"Tool {self.name} - {self.id}: started with input data {input_data}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        if self.persistent_sandbox:
            sandbox = self._sandbox
        else:
            logger.info(f"Tool {self.name} - {self.id}: Initializing Sandbox for this execution")
            sandbox = Sandbox(api_key=self.connection.api_key)
            self._install_default_packages(sandbox)
            if self.files:
                self._upload_files(files=self.files, sandbox=sandbox)
                self._update_description()
        try:
            content = {}
            if files := input_data.get("files"):
                content["files_installation"] = self._upload_files(files=files, sandbox=sandbox)
            if packages := input_data.get("packages"):
                self._install_packages(sandbox=sandbox, packages=packages)
                content["packages_installation"] = f"Installed packages: {input_data['packages']}"
            if shell_command := input_data.get("shell_command"):
                content["shell_command_execution"] = self._execute_shell_command(shell_command, sandbox=sandbox)
            if python := input_data.get("python"):
                content["code_execution"] = self._execute_python_code(python, sandbox=sandbox)
            if not (packages or files or shell_command or python):
                raise ToolExecutionException(
                    "Error: Invalid input data. Please provide 'files' for file upload (bytes or BytesIO)",
                    recoverable=True,
                )

        finally:
            if not self.persistent_sandbox:
                logger.debug(f"Tool {self.name} - {self.id}: Closing Sandbox")
                sandbox.close()

        if self.is_optimized_for_agents:
            result = ""
            if files_installation := content.get("files_installation"):
                result += "<Files installation>\n" + files_installation + "\n</Files installation>"
            if packages_installation := content.get("packages_installation"):
                result += "<Package installation>\n" + packages_installation + "\n</Package installation>"
            if shell_command_execution := content.get("shell_command_execution"):
                result += "<Shell command execution>\n" + shell_command_execution + "\n</Shell command execution>"
            if code_execution := content.get("code_execution"):
                result += "<Code execution>\n" + code_execution + "\n</Code execution>"
            content = result

        logger.debug(f"Tool {self.name} - {self.id}: finished with result {str(content)[:50]}...")
        return {"content": content}

    def close(self) -> None:
        """Closes the persistent sandbox if it exists."""
        if self._sandbox and self.persistent_sandbox:
            logger.debug(f"Tool {self.name} - {self.id}: Closing Sandbox")
            self._sandbox.close()
            self._sandbox = None
