"""Thin wrapper over the ``gh`` CLI, shared by eval.pr_bot (the periodic
gate-chain sweep) and eval.copycat_guard's real-time single-PR check (the
sensitive-paths-guard.yml-adjacent workflow) so neither has to duplicate
subprocess/GitHub plumbing.

Every method is one subprocess call, so a fake stand-in is trivial to write
for tests (see eval/tests/test_pr_bot.py's ``FakeClient``) -- this class
itself is not unit-tested, only the pure decision logic that consumes it is.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass


@dataclass
class PRInfo:
    number: int
    title: str
    author: str
    is_draft: bool
    head_sha: str
    body: str = ""
    url: str = ""
    labels: tuple[str, ...] = ()


class GitHubClient:
    def __init__(self, repo: str):
        self.repo = repo

    def _run(self, *args: str) -> str:
        result = subprocess.run(["gh", *args, "-R", self.repo],
                                capture_output=True, text=True, check=True)
        return result.stdout

    def list_prs(self, state: str = "open") -> list:
        out = self._run("pr", "list", "--state", state, "-L", "300", "--json",
                        "number,title,author,isDraft,headRefOid,body,url,labels")
        data = json.loads(out)
        return [
            PRInfo(number=d["number"], title=d["title"],
                   author=d["author"]["login"], is_draft=d["isDraft"],
                   head_sha=d["headRefOid"], body=d.get("body") or "",
                   url=d.get("url") or "",
                   labels=tuple(lbl["name"] for lbl in d.get("labels", [])))
            for d in data
        ]

    def get_pr(self, pr_number: int) -> PRInfo:
        out = self._run("pr", "view", str(pr_number), "--json",
                        "number,title,author,isDraft,headRefOid,body,url,labels")
        d = json.loads(out)
        return PRInfo(number=d["number"], title=d["title"],
                      author=d["author"]["login"], is_draft=d["isDraft"],
                      head_sha=d["headRefOid"], body=d.get("body") or "",
                      url=d.get("url") or "",
                      labels=tuple(lbl["name"] for lbl in d.get("labels", [])))

    def get_diff(self, pr_number: int) -> str:
        return self._run("pr", "diff", str(pr_number))

    def get_comments(self, pr_number: int) -> list:
        out = self._run("pr", "view", str(pr_number), "--json", "comments")
        data = json.loads(out)
        return [c["body"] for c in data.get("comments", [])]

    def post_comment(self, pr_number: int, body: str) -> None:
        subprocess.run(["gh", "pr", "comment", str(pr_number), "-R", self.repo,
                       "--body", body], check=True)

    def add_label(self, pr_number: int, label: str) -> None:
        subprocess.run(["gh", "pr", "edit", str(pr_number), "-R", self.repo,
                       "--add-label", label], check=True)

    def remove_label(self, pr_number: int, label: str) -> None:
        subprocess.run(["gh", "pr", "edit", str(pr_number), "-R", self.repo,
                       "--remove-label", label], check=False)

    def close_pr(self, pr_number: int, reason: str) -> None:
        subprocess.run(["gh", "pr", "close", str(pr_number), "-R", self.repo,
                       "--comment", reason], check=True)
