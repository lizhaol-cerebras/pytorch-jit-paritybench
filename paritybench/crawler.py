import json
import logging
import os
import re
import time
from typing import List, Optional
import requests

log = logging.getLogger(__name__)


class CrawlGitHub(object):
    """
    Download GitHub projects either by:
    1. Search query (e.g., pytorch projects with 100+ stars)
    OR
    2. Direct list of GitHub HTML URLs
    """

    def __init__(
        self,
        download_dir: str,
        max_count: Optional[int] = None,
        query: Optional[str] = None,
        html_urls: Optional[List[str]] = None,
    ):
        if query is not None and html_urls is not None:
            raise ValueError("Specify either query OR html_urls, not both")
        if query is None and html_urls is None:
            raise ValueError("Must specify either query OR html_urls")

        super(CrawlGitHub, self).__init__()
        self.download_dir = download_dir
        self.max_count = max_count  # max number of projects to download
        self.query = query
        self.html_urls = html_urls

    def parse_github_url(self, html_url: str) -> tuple:
        """
        Parse GitHub HTML URL to extract owner and repo name
        Example: https://github.com/owner/repo -> (owner, repo)
        """
        parts = html_url.rstrip("/").split("/")
        if len(parts) < 5 or parts[2] != "github.com":
            raise ValueError(f"Invalid GitHub URL: {html_url}")
        return parts[-2], parts[-1]

    def get_repo_info(self, owner: str, repo: str) -> dict:
        """
        Get repository information using GitHub API
        """
        time.sleep(6)  # Respect rate limits
        url = f"https://api.github.com/repos/{owner}/{repo}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def github_search(self):
        """
        Search GitHub repositories based on query
        """
        base = "https://api.github.com/search/repositories?per_page=100&sort=stars"
        default_query = "pytorch+language:Python+stars:>100+size:<100000"
        search_query = self.query if self.query else default_query

        seen = set()
        # both orders gets us 20 pages (past 10 limit), need 12 for current query
        for order in ("desc", "asc"):
            page = 1
            while True:
                # https://docs.github.com/en/rest/search/search?apiVersion=2022-11-28#rate-limit
                time.sleep(6)
                rs = requests.get(f"{base}&page={page}&order={order}&q={search_query}")
                rs.raise_for_status()
                result = rs.json()
                assert not result["incomplete_results"]

                for project in result["items"]:
                    name = project["full_name"]
                    if self.max_count and len(seen) >= self.max_count:
                        return
                    if name not in seen:
                        seen.add(name)
                        yield project

                total_count = result["total_count"]
                log.info(
                    f"total_count={total_count} seen={len(seen)} page={page} {order}"
                )
                page += 1
                if (
                    len(result["items"]) == 0
                    or len(seen) >= total_count
                    or (self.max_count and len(seen) >= self.max_count)
                ):
                    return
                if page == 11:
                    break  # not allowed by API

    def download_project(self, project: dict) -> str:
        """
        Download a single GitHub project
        """
        name = project["full_name"]
        url = project["html_url"]
        default_branch = project["default_branch"]
        output_filename = re.sub(r"[^a-zA-Z0-9]+", "_", name) + ".zip"
        output_path = os.path.join(self.download_dir, output_filename)

        if os.path.exists(output_path):
            return output_filename

        time.sleep(60)
        rs = requests.get(f"{url}/archive/{default_branch}.zip", stream=True)
        rs.raise_for_status()
        with open(output_path, "wb") as fd:
            for chunk in rs.iter_content(chunk_size=8192):
                fd.write(chunk)
        return output_filename

    def download(self):
        """
        Download repositories based on either search query or URL list
        """
        metadata_path = os.path.join(self.download_dir, "metadata.json")
        if os.path.exists(metadata_path):
            return

        os.path.exists(self.download_dir) or os.mkdir(self.download_dir)
        metadata = dict()

        if self.query is not None:
            # Search-based download
            projects = list(self.github_search())
            for i, project in enumerate(projects):
                log.info(
                    f"Downloading {project['full_name']} ({i + 1} of {len(projects)})"
                )
                metadata[self.download_project(project)] = project

        else:
            # URL list-based download
            for i, url in enumerate(self.html_urls):
                try:
                    owner, repo = self.parse_github_url(url)
                    project = self.get_repo_info(owner, repo)
                    log.info(
                        f"Downloading {project['full_name']} ({i + 1} of {len(self.html_urls)})"
                    )
                    metadata[self.download_project(project)] = project

                    if self.max_count and len(metadata) >= self.max_count:
                        break

                except Exception as e:
                    log.error(f"Error downloading {url}: {str(e)}")
                    continue

        with open(metadata_path, "w") as fd:
            json.dump(metadata, fd)
