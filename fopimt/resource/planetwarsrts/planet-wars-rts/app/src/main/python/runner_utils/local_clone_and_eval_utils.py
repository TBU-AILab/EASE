"""
The functions here run locally in the sense that they do not comment on GitHub issues.
"""

import os
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

from runner_utils.competition_entries import sample_entries
# from agent_entry import AgentEntry  # your model
from runner_utils.utils import run_command, find_free_port, comment_on_issue, close_issue, \
    parse_yaml_from_issue_body  # previously defined helpers
from runner_utils.agent_entry import AgentEntry  # Assuming AgentEntry is defined in agent_entry.py
import os

from util.scan_closed_issues_for_results import load_github_token

home = Path(os.path.expanduser("~"))
KOTLIN_PROJECT_PATH = home / "GitHub/planet-wars-rts/"


def process_commit_hash(agent_data: dict) -> dict:
    """
    If agent_data["repo_url"] is a full commit URL, extract the repo base and commit hash.
    Returns a new dict with normalized repo_url and optional commit field.
    """
    new_data = agent_data.copy()

    repo_url = agent_data.get("repo_url", "")
    parsed = urlparse(repo_url)
    parts = parsed.path.strip("/").split("/")

    if "commit" in parts:
        try:
            user, repo, _, commit_hash = parts[:4]
            new_data["repo_url"] = f"https://github.com/{user}/{repo}.git"
            new_data["commit"] = commit_hash
        except Exception as e:
            raise ValueError(f"Unable to parse commit URL '{repo_url}': {e}")

    return new_data



def clone_and_build_repo(agent: AgentEntry, base_dir: Path, github_token: str) -> Path | None:
    repo = "SimonLucas/planet-wars-rts-submissions"
    repo_dir = base_dir / agent.id
    gradlew_path = repo_dir / "gradlew"

    if not repo_dir.exists():
        run_command(["git", "clone", agent.repo_url, str(repo_dir)])
        print("📦 Repository cloned.")

    if agent.commit:
        run_command(["git", "checkout", agent.commit], cwd=repo_dir)
        print(f"📌 Checked out commit `{agent.commit}`")

    if not gradlew_path.exists():
        print("❌ Gradle wrapper not found in repo.")
        return None

    gradlew_path.chmod(gradlew_path.stat().st_mode | 0o111)  # Add executable bit
    run_command(["./gradlew", "build"], cwd=repo_dir)
    print("🔨 Project built successfully.", github_token)

    return repo_dir


def build_and_launch_container(agent: AgentEntry, repo_dir: Path) -> int:
    container_name = f"container-{agent.id}"
    image_name = f"game-server-{agent.id}"

    run_command(["podman", "build", "-t", image_name, "."], cwd=repo_dir)

    try:
        run_command(["podman", "rm", "-f", container_name])
    except subprocess.CalledProcessError:
        pass

    port = find_free_port()
    run_command([
        "podman", "run", "-d",
        "-p", f"{port}:8080",
        "--name", container_name,
        image_name
    ])
    print(f"🚀 Agent launched at external port `{port}`.")
    return port


def run_evaluation(port: int, timeout_seconds: int = 300) -> bool:

    print(f"🎮 Running evaluation matches...")
    try:
        subprocess.run(
            ["./gradlew", "runEvaluation", f"--args={port}"],
            cwd=KOTLIN_PROJECT_PATH,
            check=True,
            timeout=timeout_seconds,
        )
        return True
    except subprocess.TimeoutExpired:
        print(f"⏰ Evaluation timed out after {timeout_seconds}s.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed: {e}")

    return False



def stop_and_cleanup_container(agent_id: str) -> None:
    container_name = f"container-{agent_id}"
    run_command(["podman", "stop", container_name])
    run_command(["podman", "rm", container_name])
    print("✅ Evaluation complete. Stopping container.")


if __name__ == "__main__":
    # specify particular agent data to process
    # This is just a placeholder for testing purposes
    agent_entry = sample_entries[0]
    github_token = load_github_token()
    base_dir = Path("/tmp/simonl-planetwars-run")
    repo_dir = clone_and_build_repo(agent_entry, base_dir, github_token)
    if not repo_dir:
        print("❌ Failed to clone or build the repository.")
    else:
        port = build_and_launch_container(agent_entry, repo_dir)
        if port:
            if run_evaluation(port):
                print("✅ Evaluation completed successfully.")
            else:
                print("❌ Evaluation failed.")
            stop_and_cleanup_container(agent_entry.id)
        else:
            print("❌ Failed to launch container.")

