"""
Sometimes we clone a repo then the run fails somehow, so we want to
 be able to run the same repo again without cloning it again.
"""

import os
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

# from agent_entry import AgentEntry  # your model
from runner_utils.utils import run_command, find_free_port, comment_on_issue, close_issue, \
    parse_yaml_from_issue_body  # previously defined helpers
from runner_utils.agent_entry import AgentEntry  # Assuming AgentEntry is defined in agent_entry.py

home = Path(os.path.expanduser("~"))
KOTLIN_PROJECT_PATH = home / "GitHub/planet-wars-rts/"


def run_cloned_repo(base_dir: Path, agent_id: str, timeout_seconds: int = 600):
    repo_dir = base_dir / agent_id
    print(f"Repo dir: {repo_dir.exists()}")
    for path in list(repo_dir.iterdir()):
        print(f"Path: {path} - Exists: {path.exists()} - Is Dir: {path.is_dir()}")

    gradlew_path = repo_dir / "gradlew"
    print("Exists:", gradlew_path.exists())
    print("Is file:", gradlew_path.is_file())
    print("Permissions:", oct(gradlew_path.stat().st_mode))

    if not gradlew_path.exists():
        print(f"❌ Gradle wrapper not found in repo: {repo_dir}. Please ensure the repository is valid.")
        return


    run_command(["./gradlew", "build"], cwd=repo_dir)

    run_command(["podman", "build", "-t", f"game-server-{agent_id}", "."], cwd=repo_dir)

    # --- Remove any previous container with the same name ---
    container_name = f"container-{agent_id}"
    try:
        run_command(["podman", "rm", "-f", container_name])
    except subprocess.CalledProcessError:
        # It's okay if the container didn't exist
        pass

    # --- Step 3: Start container with dynamic port ---
    free_port = find_free_port()

    print(f"Using free port: {free_port}")

    run_command([
        "podman", "run", "-d",
        "-p", f"{free_port}:8080",
        "--name", container_name,
        f"game-server-{agent_id}"
    ])

    print(f"🚀 Agent launched at external port `{free_port}`.")

    # --- Step 4: Run evaluation script ---
    print(f"🎮 Running evaluation matches...")

    start_time = time.time()
    try:
        subprocess.run(
            ["./gradlew", "runEvaluation", f"--args={free_port}"],
            cwd=KOTLIN_PROJECT_PATH,
            check=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        print(f"⏰ Evaluation timed out after {timeout_seconds}s.")
        run_command(["podman", "stop", container_name])
        run_command(["podman", "rm", container_name])
        return

    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed: {e}")
        return

    # --- Step 5: Read Markdown results and post ---
    # md_file = Path("path/to/kotlin/project/results/league.md")
    md_file = Path("/Users/simonl/GitHub/planet-wars-rts/app/results/sample/league.md")

    if not md_file.exists():
        print("⚠️ Evaluation completed, but results file not found.")
    else:
        markdown = md_file.read_text()
        print(f"📊 **Results:**\n\n{markdown}")

    # --- Step 6: Shut down and close ---
    run_command(["podman", "stop", f"container-{agent_id}"])
    run_command(["podman", "rm", f"container-{agent_id}"])
    print("✅ Evaluation complete. Stopping container.")


if __name__ == "__main__":
    # Example usage
    base_dir = "/tmp/simonl-planetwars-run"
    base_path = Path(base_dir)
    agent_id = "python-test"  # Replace with actual agent ID
    # agent_id = "another-sample-agent"  # Replace with actual agent ID
    timeout_seconds = 600  # Set your desired timeout

    p1 = "/tmp/simonl-planetwars-run/python-test"
    p2 = base_path / agent_id

    print(f"Comparing paths: {p1} and {p2}")
    print(f"Resolved paths equal: {Path(p1).resolve() == p2.resolve()}")

    run_cloned_repo(base_path, agent_id, timeout_seconds)
