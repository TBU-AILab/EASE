import os
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

# from agent_entry import AgentEntry  # your model
from runner_utils.utils import run_command, find_free_port, comment_on_issue, close_issue, parse_yaml_from_issue_body  # previously defined helpers
from runner_utils.agent_entry import AgentEntry  # Assuming AgentEntry is defined in agent_entry.py
import os

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

def process_issue(issue: dict, base_dir: Path, github_token: str, timeout_seconds: int = 300):
    issue_number = issue["number"]
    body = issue["body"]
    repo = "SimonLucas/planet-wars-rts-submissions"

    # --- Step 1: Parse YAML ---
    # from yaml_parser import parse_yaml_from_issue_body  # your working parser
    agent_data = parse_yaml_from_issue_body(body)
    if not agent_data:
        comment_on_issue(repo, issue_number, "❌ Could not parse submission YAML.", github_token)
        return

    agent_data = process_commit_hash(agent_data)  # Normalize commit hash if needed

    agent = AgentEntry(**agent_data)
    agent.id = agent.id.lower()

    print(f"Processing agent data: {agent}")

    # --- Step 2: Clone and build ---
    comment_on_issue(repo, issue_number, f"🔍 Processing submission for `{agent.id}`", github_token)
    repo_dir = base_dir / agent.id
    gradlew_path = repo_dir / "gradlew"

    if not repo_dir.exists():
        run_command(["git", "clone", agent.repo_url, str(repo_dir)])
        comment_on_issue(repo, issue_number, "📦 Repository cloned.", github_token)

    if agent.commit:
        run_command(["git", "checkout", agent.commit], cwd=repo_dir)
        comment_on_issue(repo, issue_number, f"📌 Checked out commit `{agent.commit}`", github_token)

    if not gradlew_path.exists():
        comment_on_issue(repo, issue_number, "❌ Gradle wrapper not found in repo.", github_token)
        return

    run_command(["./gradlew", "build"], cwd=repo_dir)
    comment_on_issue(repo, issue_number, "🔨 Project built successfully.", github_token)

    run_command(["podman", "build", "-t", f"game-server-{agent.id}", "."], cwd=repo_dir)

    # --- Remove any previous container with the same name ---
    container_name = f"container-{agent.id}"
    try:
        run_command(["podman", "rm", "-f", container_name])
    except subprocess.CalledProcessError:
        # It's okay if the container didn't exist
        pass

    # --- Step 3: Start container with dynamic port ---
    free_port = find_free_port()

    run_command([
        "podman", "run", "-d",
        "-p", f"{free_port}:8080",
        "--name", container_name,
        f"game-server-{agent.id}"
    ])

    comment_on_issue(repo, issue_number, f"🚀 Agent launched at external port `{free_port}`.", github_token)

    # --- Step 4: Run evaluation script ---
    comment_on_issue(repo, issue_number, f"🎮 Running evaluation matches...", github_token)

    start_time = time.time()
    try:
        subprocess.run(
            ["./gradlew", "runEvaluation", f"--args={free_port}"],
            cwd=KOTLIN_PROJECT_PATH,
            check=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        comment_on_issue(repo, issue_number, f"⏰ Evaluation timed out after {timeout_seconds}s.", github_token)
        run_command(["podman", "stop", container_name])
        run_command(["podman", "rm", container_name])
        return

    except subprocess.CalledProcessError as e:
        comment_on_issue(repo, issue_number, f"❌ Evaluation failed: {e}", github_token)
        return

    # --- Step 5: Read Markdown results and post ---
    # md_file = Path("path/to/kotlin/project/results/league.md")
    # md_file = Path("/Users/simonl/GitHub/planet-wars-rts/app/results/sample/league.md")
    md_file = home / "GitHub/planet-wars-rts/app/results/sample/league.md"
    md_file = md_file.resolve()

    if not md_file.exists():
        comment_on_issue(repo, issue_number, "⚠️ Evaluation completed, but results file not found.", github_token)
    else:
        markdown = md_file.read_text()
        comment_on_issue(repo, issue_number, f"📊 **Results:**\n\n{markdown}", github_token)

    # --- Step 6: Shut down and close ---
    run_command(["podman", "stop", f"container-{agent.id}"])
    run_command(["podman", "rm", f"container-{agent.id}"])
    comment_on_issue(repo, issue_number, "✅ Evaluation complete. Stopping container.", github_token)

    close_issue(repo, issue_number, github_token)
