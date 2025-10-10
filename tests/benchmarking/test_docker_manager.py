# Copyright (c) Microsoft. All rights reserved.

from pathlib import Path

from eval_recipes.benchmarking.docker_manager import DockerManager


def test_docker_manager_lifecycle(tmp_path: Path) -> None:
    """Test that DockerManager builds image, starts container, and cleans up automatically."""
    log_dir = tmp_path / "logs"
    dockerfile = """FROM ubuntu:24.04
RUN echo "Hello from Docker build"
"""
    image_tag = "test-docker-manager-lifecycle"
    with DockerManager(log_dir=log_dir, dockerfile=dockerfile, image_tag=image_tag) as manager:
        # Verify container and image were created
        assert manager.container is not None
        assert manager.container_id is not None
        assert manager.actual_image_tag is not None
        assert image_tag in manager.actual_image_tag
        assert manager.container.status in ["created", "running"]

        build_log_file = log_dir / "build_image.log"
        assert build_log_file.exists()


def test_exec_command(tmp_path: Path) -> None:
    """Test that exec_command executes commands in the container."""
    log_dir = tmp_path / "logs"

    dockerfile = """FROM ubuntu:24.04
"""
    with DockerManager(log_dir=log_dir, dockerfile=dockerfile) as manager:
        assert manager.container is not None

        exec_result, logs = manager.exec_command(
            container=manager.container, command=["echo", "Hello from exec_command"]
        )

        assert exec_result is not None
        assert exec_result.exit_code == 0
        assert isinstance(logs, str)
        assert "Hello from exec_command" in logs

        # Verify log file was created
        log_files = list(log_dir.glob("exec_echo_Hello_from_e*.log"))
        assert len(log_files) == 1


def test_copy_files_to_container(tmp_path: Path) -> None:
    """Test that copy_files_to_container copies files correctly."""
    log_dir = tmp_path / "logs"

    dockerfile = """FROM ubuntu:24.04
"""
    with DockerManager(log_dir=log_dir, dockerfile=dockerfile) as manager:
        assert manager.container is not None

        # Prepare test files
        files = {
            "test.txt": b"Hello World",
            "script.sh": b"#!/bin/bash\necho 'Script executed'",
        }
        executable_files = {"script.sh"}

        # Copy files to container
        manager.copy_files_to_container(
            container=manager.container, files=files, dest_path="/tmp", executable_files=executable_files
        )

        # Verify files were copied
        result = manager.read_file_from_container(manager.container, "/tmp/test.txt")
        assert result == b"Hello World"

        # Verify executable file
        exec_result, logs = manager.exec_command(manager.container, ["bash", "/tmp/script.sh"])
        assert exec_result.exit_code == 0
        assert "Script executed" in logs


def test_read_file_from_container(tmp_path: Path) -> None:
    """Test that read_file_from_container reads files correctly."""
    log_dir = tmp_path / "logs"

    dockerfile = """FROM ubuntu:24.04
"""
    with DockerManager(log_dir=log_dir, dockerfile=dockerfile) as manager:
        assert manager.container is not None

        # Create a test file in container
        test_content = "Test file content"
        manager.exec_command(manager.container, ["bash", "-c", f"echo '{test_content}' > /tmp/testfile.txt"])

        # Read the file
        result = manager.read_file_from_container(manager.container, "/tmp/testfile.txt")
        assert result is not None
        assert test_content in result.decode("utf-8")

        # Test reading non-existent file
        result = manager.read_file_from_container(manager.container, "/tmp/nonexistent.txt")
        assert result is None


def test_container_env_variables(tmp_path: Path) -> None:
    """Test that container environment variables are set correctly."""
    log_dir = tmp_path / "logs"

    dockerfile = """FROM ubuntu:24.04
"""
    with DockerManager(log_dir=log_dir, dockerfile=dockerfile, container_env={"TEST_VAR": "test_value"}) as manager:
        assert manager.container is not None

        # Verify environment variable is set
        _exec_result, logs = manager.exec_command(manager.container, ["bash", "-c", "echo $TEST_VAR"])
        assert "test_value" in logs
