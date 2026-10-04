import argparse
import hashlib
import json
import logging
import re
import tempfile
import time
import xml.etree.ElementTree as ElementTree
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

from college_knowledge import (
    college_source_statistics,
    record_fetch_failure,
    upsert_college_source,
)
from database import SessionLocal
from document_processor import extract_text_from_file


BASE_URL = "https://sairam.edu.in/"
ALLOWED_HOSTS = {"sairam.edu.in", "www.sairam.edu.in"}
USER_AGENT = "EduGuideCollegeKnowledgeBot/1.0 (+https://sairam.edu.in/)"
RELEVANT_TERMS = {
    "about",
    "academ",
    "admission",
    "admissions",
    "alumni",
    "calendar",
    "campus",
    "career",
    "contact",
    "department",
    "departments",
    "faculty",
    "facility",
    "facilities",
    "fee",
    "fees",
    "innovation",
    "laborator",
    "library",
    "placement",
    "program",
    "research",
    "regulation",
    "scholarship",
    "student",
    "training",
    "undergraduate",
}
BLOCKED_PATH_TERMS = {
    "admin",
    "login",
    "logout",
    "portal",
    "wp-json",
    "wp-admin",
    "student-login",
}
SITEMAP_PATHS = (
    "/sitemap.xml",
    "/sitemap_index.xml",
    "/wp-sitemap.xml",
)
MAX_PDF_BYTES = 15 * 1024 * 1024
MAX_PAGE_BYTES = 20 * 1024 * 1024
logger = logging.getLogger("eduguide.college_ingestion")


@dataclass
class CrawledPage:
    url: str
    title: str
    content: str
    source_type: str
    retrieved_at: str
    content_hash: str


def canonicalize_url(url: str) -> str | None:
    try:
        parsed = urlsplit(url)
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"}:
        return None
    host = (parsed.hostname or "").lower()
    if host not in ALLOWED_HOSTS or parsed.username or parsed.password:
        return None
    if parsed.query:
        return None
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if not path.startswith("/"):
        path = f"/{path}"
    path = path.rstrip("/") or "/"
    return urlunsplit(("https", host, path, "", ""))


def is_relevant_url(url: str, anchor_text: str = "") -> bool:
    parsed = urlsplit(url)
    lowered_path = parsed.path.lower()
    if any(term in lowered_path for term in BLOCKED_PATH_TERMS):
        return False
    searchable = f"{lowered_path} {anchor_text.lower()}"
    return url == canonicalize_url(BASE_URL) or any(
        term in searchable for term in RELEVANT_TERMS
    )


def extract_html_content(html: str, source_url: str):
    soup = BeautifulSoup(html, "html.parser")
    title = ""
    if soup.title:
        title = soup.title.get_text(" ", strip=True)
    main = soup.find("main") or soup.find("article")
    if main is None:
        main = soup.find(id=re.compile(r"content|main", re.I))
    if main is None:
        main = soup.find(class_=re.compile(r"content|main", re.I))
    if main is None:
        main = soup.body or soup
    for node in main.select(
        "script, style, noscript, nav, footer, header, aside, "
        "[aria-hidden='true'], [role='navigation'], form, button"
    ):
        node.decompose()

    lines = []
    for node in main.find_all(
        ["h1", "h2", "h3", "h4", "p", "li", "td", "th", "dt", "dd"]
    ):
        text = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
        if text and (not lines or text != lines[-1]):
            lines.append(text)
    if not lines:
        text = re.sub(r"\s+", " ", main.get_text(" ", strip=True)).strip()
        if text:
            lines.append(text)
    if not title:
        title = source_url
    return title[:512], "\n".join(lines).strip()


class CollegeWebsiteCrawler:
    def __init__(
        self,
        *,
        max_pages: int = 100,
        delay_seconds: float = 1.0,
        timeout_seconds: float = 15.0,
        include_pdfs: bool = False,
        max_pdf_bytes: int = MAX_PDF_BYTES,
        client=None,
    ):
        if max_pages < 1:
            raise ValueError("max_pages must be at least 1.")
        if delay_seconds < 0:
            raise ValueError("delay_seconds cannot be negative.")
        self.max_pages = max_pages
        self.delay_seconds = delay_seconds
        self.timeout_seconds = timeout_seconds
        self.include_pdfs = include_pdfs
        self.max_pdf_bytes = max_pdf_bytes
        self.client = client or httpx.Client(
            timeout=timeout_seconds,
            headers={"User-Agent": USER_AGENT},
        )
        self._owns_client = client is None
        self._last_request_at = 0.0
        self.failures: list[dict[str, str]] = []
        self.robots_by_host: dict[str, RobotFileParser | None] = {}
        self.sitemaps: list[str] = []

    def close(self):
        if self._owns_client:
            self.client.close()

    def _request(self, url: str):
        wait = self.delay_seconds - (time.monotonic() - self._last_request_at)
        if wait > 0:
            time.sleep(wait)
        response = self.client.get(url, follow_redirects=False)
        self._last_request_at = time.monotonic()
        return response

    def load_robots(self):
        for host in sorted(ALLOWED_HOSTS):
            robots_url = f"https://{host}/robots.txt"
            try:
                response = self._request(robots_url)
            except httpx.HTTPError as error:
                self.failures.append(
                    {"url": robots_url, "error": type(error).__name__}
                )
                logger.warning(
                    "robots.txt request failed for %s (%s).",
                    host,
                    type(error).__name__,
                )
                self.robots_by_host[host] = None
                continue
            if response.status_code == 404:
                parser = RobotFileParser(robots_url)
                parser.parse([])
                self.robots_by_host[host] = parser
                continue
            if response.status_code != 200:
                self.failures.append(
                    {"url": robots_url, "error": f"http_{response.status_code}"}
                )
                logger.warning(
                    "robots.txt returned HTTP %s for %s; that host is excluded.",
                    response.status_code,
                    host,
                )
                self.robots_by_host[host] = None
                continue
            parser = RobotFileParser(robots_url)
            parser.parse(response.text.splitlines())
            self.robots_by_host[host] = parser
            self.sitemaps.extend(parser.site_maps() or [])
        return self.robots_by_host.get("sairam.edu.in") is not None

    def _allowed_by_robots(self, url: str) -> bool:
        host = (urlsplit(url).hostname or "").lower()
        parser = self.robots_by_host.get(host)
        if parser is None:
            return False
        return parser.can_fetch(USER_AGENT, url)

    def _get_same_domain(self, url: str):
        current_url = url
        for _ in range(5):
            if not self._allowed_by_robots(current_url):
                raise PermissionError("robots_disallow")
            response = self._request(current_url)
            if response.is_redirect:
                location = response.headers.get("location")
                redirect_url = urljoin(current_url, location or "")
                redirected = canonicalize_url(redirect_url)
                if not redirected:
                    raise ValueError("redirect_outside_approved_domain")
                if redirected == canonicalize_url(current_url):
                    source_path = urlsplit(current_url).path
                    target_path = urlsplit(redirect_url).path
                    if (
                        source_path.rstrip("/") == target_path.rstrip("/")
                        and source_path != target_path
                    ):
                        redirected_parts = urlsplit(redirected)
                        current_url = urlunsplit(
                            (
                                "https",
                                redirected_parts.netloc,
                                target_path,
                                "",
                                "",
                            )
                        )
                        continue
                    raise ValueError("redirect_loop")
                current_url = redirected
                continue
            response.raise_for_status()
            final_url = canonicalize_url(str(response.url))
            if not final_url:
                raise ValueError("response_outside_approved_domain")
            return response, final_url
        raise ValueError("too_many_redirects")

    def _sitemap_urls(self, sitemap_url: str, seen=None, budget=1000):
        if seen is None:
            seen = set()
        canonical = canonicalize_url(sitemap_url)
        if not canonical or canonical in seen or len(seen) >= budget:
            return []
        seen.add(canonical)
        try:
            response, _ = self._get_same_domain(canonical)
            root = ElementTree.fromstring(response.content)
        except (httpx.HTTPError, ElementTree.ParseError, ValueError, PermissionError) as error:
            self.failures.append(
                {"url": canonical, "error": type(error).__name__}
            )
            logger.warning("Sitemap fetch failed for %s (%s).", canonical, type(error).__name__)
            return []
        namespace = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
        urls = []
        for element in root.findall(f".//{namespace}loc"):
            target = canonicalize_url((element.text or "").strip())
            if not target:
                continue
            if target.lower().endswith(".xml"):
                urls.extend(self._sitemap_urls(target, seen, budget))
            else:
                urls.append(target)
            if len(urls) >= budget:
                break
        return urls

    def discover(self):
        if not self.load_robots():
            return []
        sitemap_urls = self.sitemaps[:]
        sitemap_urls.extend(urljoin(BASE_URL, path) for path in SITEMAP_PATHS)
        discovered = []
        seen = set()
        for sitemap_url in sitemap_urls:
            for url in self._sitemap_urls(sitemap_url, seen=set()):
                canonical = canonicalize_url(url)
                if (
                    canonical
                    and is_relevant_url(canonical)
                    and canonical not in seen
                ):
                    discovered.append(canonical)
                    seen.add(canonical)
                if len(discovered) >= self.max_pages:
                    return discovered

        queue = deque([BASE_URL])
        while queue and len(discovered) < self.max_pages:
            url = canonicalize_url(queue.popleft())
            if not url or url in seen or not is_relevant_url(url):
                continue
            seen.add(url)
            try:
                response, final_url = self._get_same_domain(url)
                content_type = response.headers.get("content-type", "").lower()
                if "html" not in content_type:
                    continue
                title, content = extract_html_content(response.text, final_url)
                if content:
                    discovered.append(final_url)
                soup = BeautifulSoup(response.text, "html.parser")
                for anchor in soup.find_all("a", href=True):
                    candidate = canonicalize_url(urljoin(final_url, anchor["href"]))
                    label = anchor.get_text(" ", strip=True)
                    if (
                        candidate
                        and candidate not in seen
                        and is_relevant_url(candidate, label)
                    ):
                        queue.append(candidate)
            except (httpx.HTTPError, UnicodeError, ValueError, PermissionError) as error:
                self.failures.append(
                    {"url": url, "error": type(error).__name__}
                )
                logger.warning("Page discovery failed for %s (%s).", url, type(error).__name__)
        return discovered

    def fetch_pages(self, urls: list[str]):
        pages = []
        for url in urls[: self.max_pages]:
            try:
                response, final_url = self._get_same_domain(url)
                content_type = response.headers.get("content-type", "").lower()
                if len(response.content) > MAX_PAGE_BYTES:
                    raise ValueError("response_too_large")
                source_type = "html"
                if "application/pdf" in content_type or final_url.lower().endswith(".pdf"):
                    if not self.include_pdfs:
                        continue
                    if len(response.content) > self.max_pdf_bytes:
                        raise ValueError("pdf_too_large")
                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf",
                        delete=False,
                    ) as temporary:
                        temporary_path = Path(temporary.name)
                        temporary.write(response.content)
                    try:
                        content = extract_text_from_file(temporary_path)
                    finally:
                        temporary_path.unlink(missing_ok=True)
                    title = Path(urlsplit(final_url).path).name or final_url
                    source_type = "pdf"
                elif "html" in content_type:
                    title, content = extract_html_content(response.text, final_url)
                else:
                    continue
                if not content.strip():
                    raise ValueError("empty_content")
                pages.append(
                    CrawledPage(
                        url=final_url,
                        title=title,
                        content=content,
                        source_type=source_type,
                        retrieved_at=datetime.now(timezone.utc).isoformat(),
                        content_hash=hashlib.sha256(
                            content.strip().encode("utf-8")
                        ).hexdigest(),
                    )
                )
            except (
                httpx.HTTPError,
                UnicodeError,
                OSError,
                ValueError,
                PermissionError,
            ) as error:
                code = (
                    str(error)
                    if str(error) in {
                        "pdf_too_large",
                        "empty_content",
                        "robots_disallow",
                    }
                    else type(error).__name__
                )
                self.failures.append({"url": url, "error": code})
                logger.warning("Page fetch failed for %s (%s).", url, code)
        return pages


def export_review_bundle(
    pages: list[CrawledPage],
    export_dir: Path,
    failures: list[dict[str, str]] | None = None,
):
    export_dir.mkdir(parents=True, exist_ok=True)
    payload_path = export_dir / "college-pages.jsonl"
    manifest_path = export_dir / "manifest.json"
    with payload_path.open("w", encoding="utf-8") as output:
        for page in pages:
            output.write(json.dumps(asdict(page), ensure_ascii=False) + "\n")
    manifest_path.write_text(
        json.dumps(
            {
                "source": BASE_URL,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "pages": [
                    {
                        "url": page.url,
                        "title": page.title,
                        "content_hash": page.content_hash,
                        "source_type": page.source_type,
                        "characters": len(page.content),
                    }
                    for page in pages
                ],
                "failures": failures or [],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Ingest relevant public information from sairam.edu.in."
    )
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-pdfs", action="store_true")
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--stats", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args(argv)
    if args.stats:
        with SessionLocal() as db:
            print(json.dumps(college_source_statistics(db), indent=2))
        return 0

    crawler = CollegeWebsiteCrawler(
        max_pages=args.max_pages,
        delay_seconds=args.delay_seconds,
        timeout_seconds=args.timeout_seconds,
        include_pdfs=args.include_pdfs,
    )
    try:
        urls = crawler.discover()
        logger.info("Discovered %s relevant official URLs.", len(urls))
        for url in urls:
            logger.info("Discovered: %s", url)
        pages = crawler.fetch_pages(urls)
        logger.info("Fetched %s pages; %s page failures.", len(pages), len(crawler.failures))
        if args.export_dir:
            export_review_bundle(pages, args.export_dir, crawler.failures)
            logger.info("Review bundle exported to %s.", args.export_dir)
        if args.dry_run:
            logger.info("Dry run complete; no page content or failure metadata was indexed.")
            return 0
        with SessionLocal() as db:
            for failure in crawler.failures:
                failure_path = urlsplit(failure["url"]).path.lower()
                is_page = (
                    failure_path != "/robots.txt"
                    and not failure_path.endswith(".xml")
                    and canonicalize_url(failure["url"]) is not None
                )
                if is_page:
                    record_fetch_failure(db, failure["url"], failure["error"])
            for page in pages:
                try:
                    upsert_college_source(
                        db,
                        source_url=page.url,
                        page_title=page.title,
                        content_text=page.content,
                        source_type=page.source_type,
                        fetched_at=datetime.fromisoformat(page.retrieved_at),
                    )
                except Exception as error:
                    db.rollback()
                    logger.exception(
                        "Indexing failed for %s (%s).",
                        page.url,
                        type(error).__name__,
                    )
                    record_fetch_failure(db, page.url, type(error).__name__)
            logger.info("Current knowledge-base statistics: %s", college_source_statistics(db))
        return 0
    finally:
        crawler.close()


if __name__ == "__main__":
    raise SystemExit(main())
