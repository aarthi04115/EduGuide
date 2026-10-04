import os
import unittest
from unittest.mock import patch

os.environ["DATABASE_URL"] = "sqlite+pysqlite://"
os.environ["JWT_SECRET_KEY"] = "test-only-secret-with-at-least-32-bytes"
os.environ["GROQ_API_KEY"] = "test-only-not-a-real-groq-key"

import httpx
import numpy as np
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from college_knowledge import (
    college_source_statistics,
    record_fetch_failure,
    rebuild_college_index,
    retrieve_college_context,
    upsert_college_source,
)
from database import Base
from ingestion.ingest_college_website import (
    CollegeWebsiteCrawler,
    canonicalize_url,
    extract_html_content,
    is_relevant_url,
)
from models import CollegeChunk, CollegeSource


class FakeClient:
    def __init__(self, responses):
        self.responses = responses
        self.requested = []

    def get(self, url, follow_redirects=False):
        self.requested.append(url)
        response = self.responses.get(url)
        if response is None:
            return httpx.Response(
                404,
                request=httpx.Request("GET", url),
            )
        return httpx.Response(
            response[0],
            headers=response[1],
            content=response[2],
            request=httpx.Request("GET", url),
        )


class CollegeIngestionTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite+pysqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine)
        self.db = self.session_factory()

    def tearDown(self):
        self.db.close()
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    def test_url_canonicalization_and_domain_restrictions(self):
        self.assertEqual(
            canonicalize_url(
                "https://www.sairam.edu.in/admissions/?utm_source=menu#apply"
            ),
            None,
        )
        self.assertEqual(
            canonicalize_url("http://sairam.edu.in/admissions/"),
            "https://sairam.edu.in/admissions",
        )
        self.assertIsNone(canonicalize_url("https://example.com/admissions"))
        self.assertFalse(is_relevant_url("https://sairam.edu.in/student-login"))
        self.assertTrue(is_relevant_url("https://sairam.edu.in/departments/cse"))

    def test_same_page_trailing_slash_redirect_is_followed_once(self):
        robots = b"User-agent: *\nAllow: /\n"
        fake = FakeClient(
            {
                "https://sairam.edu.in/robots.txt": (
                    200,
                    {"content-type": "text/plain"},
                    robots,
                ),
                "https://www.sairam.edu.in/robots.txt": (
                    404,
                    {"content-type": "text/plain"},
                    b"",
                ),
                "https://sairam.edu.in/admissions": (
                    301,
                    {"location": "https://sairam.edu.in/admissions/"},
                    b"",
                ),
                "https://sairam.edu.in/admissions/": (
                    200,
                    {"content-type": "text/html"},
                    b"<html><body><main><p>Admissions information.</p></main></body></html>",
                ),
            }
        )
        crawler = CollegeWebsiteCrawler(
            max_pages=1,
            delay_seconds=0,
            client=fake,
        )
        try:
            crawler.load_robots()
            response, final_url = crawler._get_same_domain(
                "https://sairam.edu.in/admissions"
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(final_url, "https://sairam.edu.in/admissions")
            self.assertIn("https://sairam.edu.in/admissions/", fake.requested)
        finally:
            crawler.close()

    def test_html_extraction_preserves_content_and_removes_navigation(self):
        title, content = extract_html_content(
            """
            <html><head><title>Admissions</title></head><body>
              <nav>Home | About | Login</nav>
              <main><h1>Admissions</h1><p>Applications are submitted online.</p>
              <ul><li>Undergraduate programmes</li></ul>
              <table><tr><th>Office</th><td>Admissions Cell</td></tr></table>
              <script>ignored secret</script></main>
              <footer>Cookie preferences</footer>
            </body></html>
            """,
            "https://sairam.edu.in/admissions",
        )
        self.assertEqual(title, "Admissions")
        self.assertIn("Applications are submitted online.", content)
        self.assertIn("Undergraduate programmes", content)
        self.assertIn("Admissions Cell", content)
        self.assertNotIn("Cookie preferences", content)
        self.assertNotIn("ignored secret", content)

    def test_sitemap_discovery_is_limited_to_approved_relevant_pages(self):
        robots = (
            "User-agent: *\nAllow: /\n"
            "Sitemap: https://sairam.edu.in/sitemap.xml\n"
        )
        sitemap = b"""<?xml version="1.0"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <url><loc>https://sairam.edu.in/admissions/</loc></url>
          <url><loc>https://elsewhere.example/admissions</loc></url>
          <url><loc>https://sairam.edu.in/wp-admin/</loc></url>
        </urlset>"""
        fake = FakeClient(
            {
                "https://sairam.edu.in/robots.txt": (
                    200,
                    {"content-type": "text/plain"},
                    robots.encode(),
                ),
                "https://sairam.edu.in/sitemap.xml": (
                    200,
                    {"content-type": "application/xml"},
                    sitemap,
                ),
            }
        )
        crawler = CollegeWebsiteCrawler(
            max_pages=5,
            delay_seconds=0,
            client=fake,
        )
        try:
            urls = crawler.discover()
            self.assertIn("https://sairam.edu.in/admissions", urls)
            self.assertFalse(any("elsewhere.example" in url for url in urls))
            self.assertFalse(any("wp-admin" in url for url in urls))
        finally:
            crawler.close()

    def test_fallback_discovery_honors_robots_and_ignores_external_links(self):
        home = b"""
        <html><body><main><h1>Sri Sairam</h1>
        <a href="/departments/computer-science">Departments</a>
        <a href="/student-login">Student Login</a>
        <a href="https://external.example/admissions">Admissions elsewhere</a>
        </main></body></html>
        """
        department = b"<html><body><main><h1>Computer Science</h1><p>Department facts.</p></main></body></html>"
        robots = b"User-agent: *\nDisallow: /departments\nAllow: /\n"
        fake = FakeClient(
            {
                "https://sairam.edu.in/robots.txt": (
                    200,
                    {"content-type": "text/plain"},
                    robots,
                ),
                "https://sairam.edu.in/sitemap.xml": (
                    404,
                    {"content-type": "text/plain"},
                    b"",
                ),
                "https://sairam.edu.in/sitemap_index.xml": (
                    404,
                    {"content-type": "text/plain"},
                    b"",
                ),
                "https://sairam.edu.in/wp-sitemap.xml": (
                    404,
                    {"content-type": "text/plain"},
                    b"",
                ),
                "https://sairam.edu.in/": (
                    200,
                    {"content-type": "text/html"},
                    home,
                ),
            }
        )
        crawler = CollegeWebsiteCrawler(
            max_pages=4,
            delay_seconds=0,
            client=fake,
        )
        try:
            urls = crawler.discover()
            self.assertEqual(urls, ["https://sairam.edu.in/"])
            self.assertNotIn(
                "https://sairam.edu.in/departments/computer-science",
                fake.requested,
            )
            self.assertFalse(any("external.example" in url for url in fake.requested))
        finally:
            crawler.close()

    def test_repeat_update_hash_and_stale_failure_preserve_content(self):
        source_url = "https://sairam.edu.in/admissions"
        source_id = upsert_college_source(
            self.db,
            source_url=source_url,
            page_title="Admissions",
            content_text="Official undergraduate admission process details.",
            source_type="html",
        )
        original_chunks = self.db.scalars(
            select(CollegeChunk).where(CollegeChunk.source_id == source_id)
        ).all()
        self.assertTrue(original_chunks)
        first_chunk_ids = {chunk.id for chunk in original_chunks}

        upsert_college_source(
            self.db,
            source_url=source_url,
            page_title="Admissions updated title",
            content_text="Official undergraduate admission process details.",
            source_type="html",
        )
        self.assertEqual(
            self.db.query(CollegeChunk)
            .filter(CollegeChunk.source_id == source_id)
            .count(),
            len(first_chunk_ids),
        )
        updated_content = "Official revised admission deadlines and process."
        upsert_college_source(
            self.db,
            source_url=source_url,
            page_title="Admissions",
            content_text=updated_content,
            source_type="html",
        )
        changed_chunks = self.db.scalars(
            select(CollegeChunk).where(CollegeChunk.source_id == source_id)
        ).all()
        self.assertEqual({chunk.id for chunk in changed_chunks}, first_chunk_ids)
        self.assertIn(updated_content, changed_chunks[0].content)

        record_fetch_failure(self.db, source_url, "http_503")
        source = self.db.get(CollegeSource, source_id)
        self.assertEqual(source.status, "stale")
        self.assertEqual(source.content_text, updated_content)
        self.assertEqual(college_source_statistics(self.db)["stale"], 1)

    def test_exact_content_duplicates_do_not_duplicate_index_chunks(self):
        contents = "Published contact information and department details."
        first_id = upsert_college_source(
            self.db,
            source_url="https://sairam.edu.in/contact",
            page_title="Contact",
            content_text=contents,
            source_type="html",
        )
        duplicate_id = upsert_college_source(
            self.db,
            source_url="https://sairam.edu.in/about/contact",
            page_title="Contact information",
            content_text=contents,
            source_type="html",
        )
        self.assertNotEqual(first_id, duplicate_id)
        self.assertEqual(self.db.get(CollegeSource, duplicate_id).status, "duplicate")
        self.assertEqual(
            self.db.query(CollegeChunk)
            .filter(CollegeChunk.source_id == first_id)
            .count(),
            1,
        )
        self.assertEqual(
            self.db.query(CollegeChunk)
            .filter(CollegeChunk.source_id == duplicate_id)
            .count(),
            0,
        )

    def test_retrieval_returns_official_source_url_and_title(self):
        upsert_college_source(
            self.db,
            source_url="https://sairam.edu.in/admissions",
            page_title="Admissions",
            content_text="Official undergraduate admission procedure.",
            source_type="html",
        )
        with patch(
            "college_knowledge.create_embeddings",
            side_effect=lambda values: np.tile(
                np.array([[1.0, 0.0]], dtype="float32"),
                (len(values), 1),
            ),
        ):
            self.assertEqual(rebuild_college_index(self.db), 1)
            context, sources = retrieve_college_context(
                "undergraduate admission procedure",
                self.db,
            )
        self.assertIn("Official undergraduate admission procedure.", context)
        self.assertEqual(sources[0]["title"], "Admissions")
        self.assertEqual(
            sources[0]["url"],
            "https://sairam.edu.in/admissions",
        )
        self.assertEqual(sources[0]["source_type"], "college_website")


if __name__ == "__main__":
    unittest.main()
