# Copyright (c) Microsoft. All rights reserved.

import io
from io import BytesIO
from pathlib import Path
import re
import tarfile
import uuid

import docker
from docker.models.containers import Container, ExecResult
from docker.models.images import Image


class DockerManager:
    def __init__(
        self,
        log_dir: Path,
        dockerfile: str,
        image_tag: str | None = None,
        container_env: dict[str, str] | None = None,
        **container_kwargs,
    ):
        self.log_dir = log_dir
        self.dockerfile = dockerfile
        self.image_tag = image_tag
        self.container_env = container_env or {}
        self.container_kwargs = container_kwargs

        self.client: docker.DockerClient | None = None
        self.container: Container | None = None
        self.container_id: str | None = None
        self.actual_image_tag: str | None = None

    def __enter__(self) -> "DockerManager":
        """Enter context manager, creating Docker client, building image, and starting container."""
        self.client = docker.from_env()
        _image, _build_logs, self.actual_image_tag = self._build_image(
            dockerfile=self.dockerfile, image_tag=self.image_tag
        )
        self.container, self.container_id = self._run_container(
            image_tag=self.actual_image_tag, container_env=self.container_env, **self.container_kwargs
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, cleaning up container, image, and closing Docker client."""
        if self.container:
            self._remove_container(self.container)

        if self.actual_image_tag:
            self._remove_image(self.actual_image_tag)

        if self.client:
            self.client.close()
            self.client = None

    def _get_client(self) -> docker.DockerClient:
        """Get Docker client, raising error if not initialized."""
        if self.client is None:
            raise RuntimeError("DockerManager must be used as a context manager (with DockerManager(...) as manager:)")
        return self.client

    def _build_image(
        self, dockerfile: str, image_tag: str | None = None, log_filename: str = "build_image.log"
    ) -> tuple[Image, str, str]:
        """
        Builds a Docker image from the specified Dockerfile.

        Args:
            dockerfile: Dockerfile content as a string
            image_tag: Optional tag for the image. If None, Docker will auto-generate.
            log_filename: Name of the log file to write build logs to (default: "build_image.log")

        Returns:
            A tuple containing the built Image object, the logs from the build process, and the actual image tag.
        """
        client = self._get_client()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.log_dir / log_filename

        image, build_logs_generator = client.images.build(
            fileobj=BytesIO(dockerfile.encode()),
            tag=image_tag,
            rm=True,
        )

        # Stream logs to file and collect them
        complete_logs = ""
        with log_file.open("w") as f:
            for chunk in build_logs_generator:
                if isinstance(chunk, dict) and "stream" in chunk:
                    text = chunk["stream"]
                    if isinstance(text, str):
                        f.write(text)
                        complete_logs += text

        # Get the actual tag assigned to the image
        if image.tags:
            actual_tag = image.tags[0]
        elif image.id:
            actual_tag = image.id
        else:
            raise RuntimeError("Built image has no tags or ID")

        return (image, complete_logs, actual_tag)

    def _run_container(
        self,
        image_tag: str,
        container_env: dict[str, str] | None = None,
        **kwargs,
    ) -> tuple[Container, str]:
        """
        Runs a Docker container.

        Args:
            image_tag: The image tag to run
            container_env: Environment variables to pass to the container
            **kwargs: Additional keyword arguments to pass to client.containers.run()

        Returns:
            A tuple containing the Container object and the container ID.
        """
        client = self._get_client()
        run_params = {
            "image": image_tag,
            "environment": container_env or {},
            "detach": True,
            "tty": True,
            "stdin_open": True,
        }
        # Override with any user-provided kwargs
        run_params.update(kwargs)
        container = client.containers.run(**run_params)

        if not container:
            raise RuntimeError("Failed to create container")

        container_id = container.id
        if not container_id:
            raise RuntimeError("Container created but has no ID")

        return (container, container_id)

    def _remove_container(self, container: Container | str, force: bool = True) -> bool:
        """
        Removes a Docker container with graceful error handling.
        """
        client = self._get_client()
        try:
            container_obj = client.containers.get(container) if isinstance(container, str) else container
            container_obj.remove(force=force)
            return True
        except Exception:
            return False

    def _remove_image(self, image_tag: str, force: bool = True) -> bool:
        """
        Removes a Docker image with graceful error handling.
        """
        client = self._get_client()
        try:
            client.images.remove(image_tag, force=force)
            return True
        except Exception:
            return False

    def exec_command(
        self, container: Container, command: str | list[str], log_filename: str | None = None, **kwargs
    ) -> tuple[ExecResult, str]:
        """
        Executes a command in a running container.

        Args:
            container: The container to execute the command in
            command: Command to execute (string or list of strings)
            log_filename: Optional name for the log file. If None, auto-generates from command.
            **kwargs: Additional keyword arguments for exec_create (e.g., workdir, environment, user)
                     or exec_start (e.g., socket). Supported exec_create params: stdout, stderr, stdin,
                     tty, privileged, user, environment, workdir.

        Returns:
            A tuple containing the ExecResult and the logs from the execution.
        """
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Generate log filename if not provided
        if log_filename is None:
            command_str = " ".join(command) if isinstance(command, list) else command
            sanitized_cmd = re.sub(r"[^a-zA-Z0-9]", "_", command_str[:20])
            unique_id = str(uuid.uuid4())[:8]
            log_filename = f"exec_{sanitized_cmd}_{unique_id}.log"

        log_file = self.log_dir / log_filename

        exec_params = {
            "cmd": command,
            "stdout": True,
            "stderr": True,
            "stdin": False,
            "tty": False,
        }

        # Parameters that go to exec_create
        exec_create_keys = ["stdout", "stderr", "stdin", "tty", "privileged", "user", "environment", "workdir"]
        for key in exec_create_keys:
            if key in kwargs:
                exec_params[key] = kwargs.pop(key)

        client = self._get_client()
        exec_id = client.api.exec_create(container.id, **exec_params)["Id"]

        # Start streaming with demux - remaining kwargs go here (like 'socket')
        stream_params = {"stream": True, "demux": True}
        stream_params.update(kwargs)
        output_stream = client.api.exec_start(exec_id, **stream_params)

        # Stream output to file and collect logs
        complete_logs = ""
        with log_file.open("wb") as f:
            for chunk in output_stream:
                if chunk:
                    if isinstance(chunk, tuple):
                        # demux=True returns (stdout, stderr) tuples
                        stdout, stderr = chunk
                        if stdout:
                            f.write(stdout)
                            complete_logs += stdout.decode("utf-8", errors="ignore")
                        if stderr:
                            f.write(stderr)
                            complete_logs += stderr.decode("utf-8", errors="ignore")
                    else:
                        f.write(chunk)
                        complete_logs += chunk.decode("utf-8", errors="ignore")

        # Get exit code after stream completes
        exec_info = client.api.exec_inspect(exec_id)
        exit_code = exec_info["ExitCode"]

        # Create ExecResult-like object
        exec_result = ExecResult(exit_code=exit_code, output=complete_logs)
        return (exec_result, complete_logs)

    def copy_files_to_container(
        self,
        container: Container,
        files: dict[str, bytes],
        dest_path: str,
        executable_files: set[str] | None = None,
    ) -> None:
        """
        Copies files to a container by creating a tar archive.

        Args:
            container: Container to copy files to
            files: Dictionary mapping filename to file content (bytes)
            dest_path: Destination path in container
            executable_files: Optional set of filenames to mark as executable
        """
        executable_files = executable_files or set()

        # Create tar archive
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            for filename, content in files.items():
                tarinfo = tarfile.TarInfo(name=filename)
                tarinfo.size = len(content)
                if filename in executable_files:
                    tarinfo.mode = 0o755
                tar.addfile(tarinfo, io.BytesIO(content))
        tar_stream.seek(0)

        container.put_archive(dest_path, tar_stream)

    def read_file_from_container(self, container: Container, file_path: str) -> bytes | None:
        """
        Reads a file from a container using exec cat command.

        Returns:
            File contents as bytes, or None if file could not be read
        """
        result = container.exec_run(["cat", file_path])
        if result.exit_code == 0:
            return result.output
        return None
