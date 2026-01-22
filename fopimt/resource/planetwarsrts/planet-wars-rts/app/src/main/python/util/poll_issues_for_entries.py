from pathlib import Path

import requests
import yaml

from runner_utils.process_issue import process_issue

# Your GitHub repo and personal access token
REPO = "SimonLucas/planet-wars-rts-submissions"

from pathlib import Path


def load_github_token() -> str:
    token_file = Path.home() / ".github_submission_token"
    if not token_file.exists():
        raise FileNotFoundError(f"GitHub token file not found at {token_file}")

    token = token_file.read_text().strip()
    return token

TOKEN = load_github_token()
if not TOKEN:
    raise RuntimeError("GITHUB_TOKEN not set in environment")

print(TOKEN)

HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github+json",
}

def get_open_issues():
    url = f"https://api.github.com/repos/{REPO}/issues"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json()

import re
import yaml

def parse_yaml_from_issue_body(body):
    try:
        # Use regex to extract the yaml code block
        match = re.search(r"```yaml\s+(.*?)```", body, re.DOTALL)
        if not match:
            raise ValueError("No valid YAML block found")
        yaml_str = match.group(1).strip()
        return yaml.safe_load(yaml_str)
    except Exception as e:
        print("YAML parsing error:", e)
        return None

def main():
    issues = get_open_issues()
    base_dir = Path("/tmp/simonl-planetwars-run")

    # check whether any open issues exist, and if not, exit with a message
    if not issues:
        print("No open issues found.")
        return

    for issue in issues:
        print(f"Issue #{issue['number']}: {issue['title']}")
        process_issue(issue, base_dir, TOKEN)

if __name__ == "__main__":
    main()
