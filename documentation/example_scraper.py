#!/usr/bin/env python3
"""
Ethical Disney Careers job scraper.

Rules:
- Respects robots.txt
- Uses a clear User-Agent
- Rate-limited
- Does not bypass auth, captchas, or blocks
- Does not scrape disallowed /search-jobs/ pages
"""

import csv
import time
import json
import argparse
import urllib.robotparser
from dataclasses import dataclass, asdict
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


BASE = "https://www.disneycareers.com"
ROBOTS = f"{BASE}/robots.txt"

USER_AGENT = "EthicalJobResearchBot/1.0 contact: you@example.com"


@dataclass
class Job:
    url: str
    title: str | None = None
    company: str | None = None
    location: str | None = None
    date_posted: str | None = None
    employment_type: str | None = None
    description: str | None = None


class EthicalScraper:
    def __init__(self, delay: float = 2.0):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

        self.robots = urllib.robotparser.RobotFileParser()
        self.robots.set_url(ROBOTS)
        self.robots.read()

    def allowed(self, url: str) -> bool:
        return self.robots.can_fetch(USER_AGENT, url)

    def fetch(self, url: str) -> str | None:
        if not self.allowed(url):
            print(f"SKIP robots.txt disallows: {url}")
            return None

        time.sleep(self.delay)

        r = self.session.get(url, timeout=20)
        if r.status_code == 429:
            raise RuntimeError("Rate limited. Stop scraping and increase delay.")
        r.raise_for_status()
        return r.text

    def parse_job(self, url: str, html: str) -> Job:
        soup = BeautifulSoup(html, "html.parser")

        # Prefer JSON-LD JobPosting data when available.
        for tag in soup.select('script[type="application/ld+json"]'):
            try:
                data = json.loads(tag.string or "")
            except json.JSONDecodeError:
                continue

            items = data if isinstance(data, list) else [data]
            for item in items:
                if item.get("@type") == "JobPosting":
                    location = item.get("jobLocation")
                    if isinstance(location, dict):
                        addr = location.get("address", {})
                        location = ", ".join(
                            filter(None, [
                                addr.get("addressLocality"),
                                addr.get("addressRegion"),
                                addr.get("addressCountry"),
                            ])
                        )

                    org = item.get("hiringOrganization")
                    company = org.get("name") if isinstance(org, dict) else None

                    return Job(
                        url=url,
                        title=item.get("title"),
                        company=company,
                        location=location,
                        date_posted=item.get("datePosted"),
                        employment_type=item.get("employmentType"),
                        description=BeautifulSoup(
                            item.get("description", ""), "html.parser"
                        ).get_text(" ", strip=True),
                    )

        # Fallback parsing.
        title = soup.find(["h1", "title"])
        text = soup.get_text(" ", strip=True)

        return Job(
            url=url,
            title=title.get_text(" ", strip=True) if title else None,
            description=text[:5000],
        )

    def scrape_urls(self, urls: list[str]) -> list[Job]:
        jobs = []

        for raw_url in urls:
            url = urljoin(BASE, raw_url)
            html = self.fetch(url)
            if not html:
                continue

            jobs.append(self.parse_job(url, html))

        return jobs


def save_csv(jobs: list[Job], path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(jobs[0]).keys())
        writer.writeheader()
        for job in jobs:
            writer.writerow(asdict(job))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", nargs="+", required=True, help="Allowed Disney job-detail URLs")
    parser.add_argument("--out", default="disney_jobs.csv")
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()

    scraper = EthicalScraper(delay=args.delay)
    jobs = scraper.scrape_urls(args.urls)

    if jobs:
        save_csv(jobs, args.out)
        print(f"Saved {len(jobs)} jobs to {args.out}")
    else:
        print("No jobs scraped.")


if __name__ == "__main__":
    main()
