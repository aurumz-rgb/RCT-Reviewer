"""
PDF Parser for RCT-Reviewer v1.0.0

Parses PDF documents using GROBID (preferred) or PyMuPDF (fallback).

Copyright (C) 2024-2026 Vihaan Sahu
Based on RobotReviewer PDF parsing by Iain Marshall, Joel Kuiper, Byron Wallace
"""

import hashlib
import logging
import re
import xml.etree.cElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from typing import Any
from urllib.parse import urljoin

import requests

from rct_reviewer.config import settings
from rct_reviewer.core.exceptions import (
    GrobidError,
    GrobidNotAvailableError,
    PDFParseError,
)
from rct_reviewer.core.models import Author, DocumentAnalysis, PublicationInfo

log = logging.getLogger(__name__)

TRIAL_ID_REGEX = re.compile(
    r"((?:ACTRN|CTRI\/|ChiCTR\-|DRKS|EUCTR|IRCT|ISRCTN|JPRN\-|KCT|NCT|RBR\-|RPCEC|TCTR)"
    r"[0-9a-zA-Z\-\/]+)"
)

TEI_NS = "{http://www.tei-c.org/ns/1.0}"


@dataclass
class ParsedPDF:
    """Raw parsed PDF data before conversion to DocumentAnalysis."""
    title: str | None = None
    abstract: str | None = None
    full_text: str | None = None
    authors: list[Author] = field(default_factory=list)
    journal: str | None = None
    year: int | None = None
    month: int | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    page_from: str | None = None
    page_to: str | None = None
    trial_registry_id: str | None = None
    file_hash: str | None = None
    parse_error: bool = False
    parser_used: str = "unknown"


class GrobidClient:
    """Client for GROBID TEI parsing service."""

    def __init__(self, host=None, timeout=None, max_retries=None):
        self.host = (host or settings.grobid.host).rstrip("/")
        self.timeout = timeout or settings.grobid.timeout
        self.max_retries = max_retries or settings.grobid.max_retries
        self._available = None
        self._session = requests.Session()

    @property
    def process_url(self):
        return urljoin(self.host, "/api/processFulltextDocument")

    def is_available(self):
        if self._available is not None:
            return self._available
        try:
            response = self._session.get(self.host, timeout=10)
            self._available = response.status_code == 200
        except requests.RequestException:
            self._available = False
        if not self._available:
            log.warning("GROBID not available at %s", self.host)
        return self._available

    def parse(self, pdf_bytes):
        if not self.is_available():
            raise GrobidNotAvailableError(self.host)

        files = {"input": ("document.pdf", pdf_bytes, "application/pdf")}
        params = {
            "consolidateHeader": "1",
            "consolidateCitations": "0",
            "consolidateSections": "1",
        }

        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self._session.post(
                    self.process_url,
                    files=files,
                    params=params,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.text
            except requests.Timeout as e:
                last_error = e
                log.warning("GROBID timeout (attempt %d/%d)", attempt + 1, self.max_retries)
            except requests.HTTPError as e:
                last_error = e
                status = e.response.status_code if e.response is not None else None
                if status == 500:
                    log.warning("GROBID 500 error (attempt %d/%d)", attempt + 1, self.max_retries)
                else:
                    raise GrobidError(str(e), status_code=status)
            except requests.RequestException as e:
                last_error = e
                log.warning("GROBID request error (attempt %d/%d)", attempt + 1, self.max_retries)

        raise GrobidError("Failed after %d retries: %s" % (self.max_retries, last_error))


class PyMuPDFParser:
    """Fallback PDF parser using PyMuPDF."""

    def __init__(self):
        try:
            import fitz
            self._fitz = fitz
        except ImportError:
            raise PDFParseError("PyMuPDF (fitz) not installed. Install with: pip install pymupdf")

    def parse(self, pdf_bytes):
        if isinstance(pdf_bytes, bytes):
            pdf_bytes = BytesIO(pdf_bytes)

        try:
            doc = self._fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            raise PDFParseError("Failed to open PDF: %s" % e)

        full_text_parts = []
        metadata = doc.metadata

        for page in doc:
            text = page.get_text("text")
            if text.strip():
                full_text_parts.append(text)

        doc.close()

        full_text = "\n".join(full_text_parts)

        title = metadata.get("title") or ""
        if not title and full_text:
            first_lines = [l.strip() for l in full_text.split("\n") if l.strip()]
            if first_lines:
                title = first_lines[0][:200]

        abstract = self._extract_abstract(full_text)
        trial_id = self._extract_trial_id(full_text)

        return ParsedPDF(
            title=title or None,
            abstract=abstract or None,
            full_text=full_text or None,
            journal=metadata.get("journal") or None,
            year=self._parse_year(metadata.get("creationDate")),
            pages=metadata.get("pages") or None,
            trial_registry_id=trial_id,
            parse_error=False,
            parser_used="pymupdf",
        )

    def _extract_abstract(self, text):
        patterns = [
            r"(?:Abstract|ABSTRACT|Background)\s*[:\-]\s*(.*?)(?:\n\s*\n|Introduction|Methods|INTRODUCTION|METHODS)",
            r"(?:Abstract|ABSTRACT)\s*\n(.*?)(?:\n\s*\n|Keywords|KEYWORDS|Introduction|METHODS)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                abstract = match.group(1).strip()
                if len(abstract) > 50:
                    return abstract
        return None

    def _extract_trial_id(self, text):
        match = TRIAL_ID_REGEX.search(text)
        return match.group(1) if match else None

    def _parse_year(self, date_str):
        if not date_str:
            return None
        match = re.search(r"(\d{4})", date_str)
        return int(match.group(1)) if match else None


class TEIParser:
    """Parser for GROBID TEI-XML output."""

    def parse(self, tei_xml):
        result = ParsedPDF(parser_used="grobid")
        full_text_parts = []
        path = []

        try:
            for event, elem in ET.iterparse(BytesIO(tei_xml.encode("utf-8")), events=("start", "end")):
                if event == "start":
                    path.append(elem.tag)
                elif event == "end":
                    if elem.tag == TEI_NS + "abstract":
                        result.abstract = self._extract_text(elem)
                    elif elem.tag == TEI_NS + "title" and TEI_NS + "titleStmt" in path:
                        result.title = self._extract_text(elem)
                    elif elem.tag in (TEI_NS + "head", TEI_NS + "p"):
                        text = self._extract_text(elem)
                        if text:
                            full_text_parts.append(text)
                    elif elem.tag == TEI_NS + "persName" and TEI_NS + "fileDesc" in path:
                        author = self._parse_author(elem)
                        if author:
                            result.authors.append(author)
                    elif elem.tag == TEI_NS + "date" and elem.attrib.get("type") == "published" and TEI_NS + "fileDesc" in path:
                        date_str = elem.attrib.get("when")
                        if date_str:
                            try:
                                parsed_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                                result.year = parsed_date.year
                                result.month = parsed_date.month
                            except (ValueError, TypeError):
                                pass
                    elif elem.tag == TEI_NS + "biblScope" and TEI_NS + "fileDesc" in path:
                        unit = elem.attrib.get("unit")
                        if unit == "volume":
                            result.volume = elem.text
                        elif unit == "issue":
                            result.issue = elem.text
                        elif unit == "page":
                            result.page_from = elem.attrib.get("from")
                            result.page_to = elem.attrib.get("to")
                            if result.page_from and result.page_to:
                                result.pages = "%s-%s" % (result.page_from, result.page_to)
                    elif elem.tag == TEI_NS + "title" and TEI_NS + "fileDesc" in path:
                        result.journal = elem.text

                    if len(path) > 0:
                        path.pop()
        except ET.ParseError as e:
            log.error("TEI-XML parse error: %s", e)
            result.parse_error = True

        result.full_text = "\n".join(full_text_parts) if full_text_parts else None

        if result.full_text:
            match = TRIAL_ID_REGEX.search(result.full_text)
            if match:
                result.trial_registry_id = match.group(1)

        return result

    def _extract_text(self, elem):
        try:
            parts = ET.tostringlist(elem, method="text", encoding="utf-8")
            return " ".join(
                p.decode("utf-8") if isinstance(p, bytes) else p
                for p in parts
                if p is not None
            ).strip()
        except Exception:
            return elem.text or ""

    def _parse_author(self, elem):
        forenames = [e.text for e in elem.findall(TEI_NS + "forename") if e.text]
        lastnames = [e.text for e in elem.findall(TEI_NS + "surname") if e.text]
        if not forenames and not lastnames:
            return None
        initials = [f[0] for f in forenames if f]
        return Author(
            initials="".join(initials),
            forename=" ".join(forenames),
            lastname=" ".join(lastnames),
        )


class PDFParser:
    """Main PDF parser that uses GROBID with PyMuPDF fallback."""

    def __init__(self, grobid_client=None, use_grobid=None):
        self.grobid = grobid_client or GrobidClient()
        self.use_grobid = use_grobid if use_grobid is not None else settings.grobid.enabled
        self.tei_parser = TEIParser()
        self._pymupdf = None

    @property
    def pymupdf(self):
        if self._pymupdf is None:
            self._pymupdf = PyMuPDFParser()
        return self._pymupdf

    def compute_hash(self, pdf_bytes):
        return hashlib.sha256(pdf_bytes).hexdigest()

    def parse(self, pdf_bytes, filename=""):
        file_hash = self.compute_hash(pdf_bytes)
        parsed = None

        if self.use_grobid and self.grobid.is_available():
            try:
                log.debug("Using GROBID to parse: %s", filename)
                tei_xml = self.grobid.parse(pdf_bytes)
                parsed = self.tei_parser.parse(tei_xml)
                parsed.file_hash = file_hash
            except GrobidError as e:
                log.warning("GROBID failed for %s: %s, falling back to PyMuPDF", filename, e)
                parsed = self._parse_with_pymupdf(pdf_bytes, file_hash)
        else:
            if self.use_grobid:
                log.info("GROBID not available, using PyMuPDF fallback")
            parsed = self._parse_with_pymupdf(pdf_bytes, file_hash)

        return self._to_document_analysis(parsed, filename)

    def _parse_with_pymupdf(self, pdf_bytes, file_hash):
        try:
            parsed = self.pymupdf.parse(pdf_bytes)
            parsed.file_hash = file_hash
            return parsed
        except PDFParseError as e:
            log.error("PyMuPDF parsing failed: %s", e)
            return ParsedPDF(
                file_hash=file_hash,
                parse_error=True,
                parser_used="failed",
            )

    def _to_document_analysis(self, parsed, filename):
        pub_info = PublicationInfo(
            title=parsed.title,
            abstract=parsed.abstract,
            journal=parsed.journal,
            year=parsed.year,
            month=parsed.month,
            volume=parsed.volume,
            issue=parsed.issue,
            pages=parsed.pages,
            authors=parsed.authors,
            trial_registry_id=parsed.trial_registry_id,
        )

        warnings = []
        if parsed.parser_used == "pymupdf":
            warnings.append("PDF parsed with PyMuPDF fallback (GROBID unavailable)")

        return DocumentAnalysis(
            filename=filename,
            file_hash=parsed.file_hash,
            full_text=parsed.full_text,
            publication=pub_info,
            parse_error=parsed.parse_error,
            models_used=["pdf_parser:" + parsed.parser_used],
            warnings=warnings,
        )

    def parse_batch(self, pdf_files):
        results = []
        for pdf_bytes, filename in pdf_files:
            try:
                doc = self.parse(pdf_bytes, filename)
                results.append(doc)
            except Exception as e:
                log.error("Failed to parse %s: %s", filename, e)
                results.append(
                    DocumentAnalysis(
                        filename=filename,
                        parse_error=True,
                        warnings=[str(e)],
                    )
                )
        return results


def parse_pdf(pdf_bytes, filename=""):
    parser = PDFParser()
    return parser.parse(pdf_bytes, filename)


def parse_pdfs(pdf_files):
    parser = PDFParser()
    return parser.parse_batch(pdf_files)