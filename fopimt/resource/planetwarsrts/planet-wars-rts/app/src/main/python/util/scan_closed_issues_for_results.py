from github import Github, Issue, IssueComment
import yaml
import re
import os
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional, Tuple, List, Dict


def load_github_token() -> str:
    token_file = Path.home() / ".github_submission_token"
    if not token_file.exists():
        raise FileNotFoundError(f"GitHub token file not found at {token_file}")
    return token_file.read_text().strip()


def extract_yaml_from_issue_body(body: str) -> Dict[str, str]:
    match = re.search(r"```yaml\n(.*?)\n```", body, re.DOTALL)
    if not match:
        print("⚠️ No YAML block found.")
        return {}
    try:
        return yaml.safe_load(match.group(1))
    except yaml.YAMLError as e:
        print(f"⚠️ YAML parsing error: {e}")
        return {}


def extract_results_from_comment(body: str) -> Tuple[Optional[float], str]:
    # Normalize newlines
    body = body.replace('\r\n', '\n')

    # Search for AVG=<score> anywhere in the text
    avg_match = re.search(r"AVG\s*=?\s*([0-9.]+)", body, re.IGNORECASE)
    if not avg_match:
        print("⚠️ AVG=... not found in comment.")
        return None, ""

    avg_score = float(avg_match.group(1))

    # Extract from "Results" onward to AVG= line
    result_block_match = re.search(
        r"(results[:：]?.*?AVG\s*=?\s*[0-9.]+)",
        body,
        re.DOTALL | re.IGNORECASE
    )
    if not result_block_match:
        print("⚠️ Could not extract full results block.")
        return avg_score, ""

    full_results = result_block_match.group(1).strip()
    return avg_score, full_results


def parse_commit_from_url(url: str) -> str:
    match = re.search(r"/commit/([a-f0-9]{7,40})$", url)
    return match.group(1) if match else "HEAD"


def extract_entry_id(yaml_data: Dict[str, str], fallback_url: str) -> str:
    if "id" in yaml_data:
        return yaml_data["id"]
    return urlparse(fallback_url).path.strip("/").split("/")[-1]




def find_result_comment(issue: Issue.Issue) -> Optional[str]:
    for comment in issue.get_comments():
        body = comment.body.lower()
        print(f"\n--- Comment from issue #{issue.number} ---")
        print(body.encode("unicode_escape").decode("ascii"))
        print("--- End comment ---\n")

        # Match if it includes any variant of "results", with or without emoji or formatting
        if "results" in body and "avg=" in body:
            return comment.body

    return None



def process_issue(issue: Issue.Issue) -> Optional[Tuple[str, str, float, str]]:
    print(f"🔍 Processing issue #{issue.number}: {issue.title}")

    yaml_data = extract_yaml_from_issue_body(issue.body)
    if not yaml_data:
        return None

    result_comment = find_result_comment(issue)
    if not result_comment:
        print("⚠️ No result comment found.")
        return None
    else:
        print(f"<UNK> Result comment from issue #{issue.number}: {result_comment}")

    avg_score, full_results = extract_results_from_comment(result_comment)
    if avg_score is None:
        return None

    repo_url = yaml_data.get("repo_url")
    if not repo_url:
        print("⚠️ No repo_url in YAML.")
        return None

    entry_id = extract_entry_id(yaml_data, repo_url)
    commit_hash = parse_commit_from_url(repo_url)

    return entry_id, commit_hash, avg_score, full_results


def generate_league_table(repo_name: str, token: str) -> str:
    g = Github(token)
    repo = g.get_repo(repo_name)
    closed_issues = repo.get_issues(state="closed")

    league_entries: List[Tuple[str, str, float]] = []
    full_results_blocks: List[str] = []

    for issue in closed_issues:
        result = process_issue(issue)
        if result:
            entry_id, commit_hash, avg_score, full_results = result
            league_entries.append((entry_id, commit_hash, avg_score))
            full_results_blocks.append(
                f"\n### {entry_id} (`{commit_hash}`)\n\n```\n{full_results}\n```"
            )

    # Sort by score descending
    league_entries.sort(key=lambda x: x[2], reverse=True)

    # Markdown league table
    league_table = "| Entry | Commit | Average Score |\n|---|---|---|\n"
    for entry_id, commit_hash, score in league_entries:
        league_table += f"| {entry_id} | `{commit_hash}` | {score:.1f} |\n"

    full_doc = f"# 🏆 PlanetWars League Table\n\n{league_table}\n\n---\n\n## 📋 Full Results\n" + "\n".join(full_results_blocks)
    return full_doc


if __name__ == "__main__":
    GITHUB_TOKEN = load_github_token()
    REPO_NAME = "SimonLucas/planet-wars-rts-submissions"

    markdown_output = generate_league_table(REPO_NAME, GITHUB_TOKEN)

    output_path = Path(__file__).resolve().parent.parent.parent.parent.parent.parent / "results" / "issues_league_table.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # output_path = "league_table.md"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_output)

    print(f"✅ League table written to {output_path}")
