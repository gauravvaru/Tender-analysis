"""
Document ingestion, OCR, and section detection
"""
import logging
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Generator

import pdfplumber
import pytesseract
from PIL import Image
import pandas as pd

from config import (
    OCR_ENABLED, EXTRACT_TABLES, MIN_TEXT_LENGTH,
    CHUNK_SIZE, CHUNK_OVERLAP
)

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process PDFs: extract text, tables, sections, metadata"""

    SECTION_KEYWORDS = {
        "scope_of_work": [
            r"scope\s+of\s+work", r"scope\s+of\s+services",
            r"work\s+description", r"project\s+scope", r"deliverables"
        ],
        "technical_specifications": [
            r"technical\s+specification", r"technical\s+requirement",
            r"technical\s+scope", r"specifications", r"technical\s+specs"
        ],
        "bill_of_quantities": [
            r"bill\s+of\s+quantities", r"\bboq\b", r"schedule\s+of\s+items",
            r"schedule\s+of\s+rates"
        ],
        "eligibility_criteria": [
            r"eligibility", r"pre-qualification", r"bidder\s+qualification",
            r"qualification\s+criteria"
        ],
        "terms_conditions": [
            r"terms\s+and\s+conditions", r"terms\s+&\s+conditions",
            r"contract\s+terms", r"general\s+conditions"
        ],
        "evaluation_criteria": [
            r"evaluation\s+criteria", r"selection\s+criteria",
            r"award\s+criteria", r"scoring\s+criteria"
        ]
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public: standard (full-file) processing
    # ------------------------------------------------------------------

    def process_file(self, file_path: str) -> Dict:
        """
        Main entry point: process a document in one pass.

        Returns:
            {
                "status":     "success" | "failed",
                "file_hash":  str,
                "content":    str,
                "metadata":   dict,
                "sections":   List[Dict],
                "chunks":     List[Dict],
                "page_count": int,
                "error":      str  (only if failed)
            }
        """
        try:
            self.logger.info(f"Processing: {file_path}")
            file_hash = self._compute_hash(file_path)

            if file_path.endswith(".pdf"):
                content_result = self._extract_from_pdf(file_path)
            else:
                content_result = self._extract_from_image(file_path)

            if content_result["status"] != "success":
                return content_result

            content    = content_result["content"]
            page_count = content_result["page_count"]

            sections = self._detect_sections(content)
            metadata = self._extract_basic_metadata(content)
            chunks   = self._create_chunks(content, sections)

            self.logger.info(
                f"Processing complete: {len(chunks)} chunks, {len(sections)} sections"
            )

            return {
                "status":     "success",
                "file_hash":  file_hash,
                "content":    content,
                "metadata":   metadata,
                "sections":   sections,
                "chunks":     chunks,
                "page_count": page_count,
            }

        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            return {"status": "failed", "error": str(e)}

    # ------------------------------------------------------------------
    # Public: streaming processing for large PDFs (500-600 pages)
    # ------------------------------------------------------------------

    def process_file_streaming(
        self, file_path: str, batch_size: int = 50
    ) -> Generator[Dict, None, None]:
        """
        Generator version of process_file for very large PDFs.

        Yields batches of chunks so the caller can embed and store them
        incrementally, keeping memory flat regardless of PDF size.

        Usage in app.py:
            for batch in processor.process_file_streaming(path, batch_size=50):
                if batch["status"] != "success":
                    break
                embeddings = embedder.embed_texts(
                    [c["text"] for c in batch["chunks"]]
                )
                store.add_vectors(embeddings, batch["chunks"])

        Yields dicts:
            {
                "status":     "success" | "failed",
                "chunks":     List[Dict],
                "page_from":  int,
                "page_to":    int,
                "page_count": int,   # total pages (first yield only)
                "file_hash":  str,   # (first yield only)
                "error":      str    # only on failure
            }
        """
        try:
            file_hash = self._compute_hash(file_path)

            with pdfplumber.open(file_path) as pdf:
                total_pages: int        = len(pdf.pages)
                sections: List[Dict]    = []
                chunk_id: int           = 0
                pending_text: str       = ""
                page_buffer: List[int]  = []
                first_yield: bool       = True

                for page_num, page in enumerate(pdf.pages, 1):
                    text          = page.extract_text() or ""
                    pending_text += f"--- PAGE {page_num} ---\n{text}\n"
                    page_buffer.append(page_num)

                    # Detect sections lazily once we have enough text
                    if not sections and len(pending_text) > 2000:
                        sections = self._detect_sections(pending_text)

                    # Emit a batch every batch_size pages, and on the final page
                    if len(page_buffer) >= batch_size or page_num == total_pages:
                        chunks = self._create_chunks(pending_text, sections)

                        for c in chunks:
                            c["file_hash"]   = file_hash
                            c["source_file"] = Path(file_path).name
                            c["id"]          = chunk_id
                            c["chunk_index"] = chunk_id
                            chunk_id        += 1

                        payload: Dict = {
                            "status":    "success",
                            "chunks":    chunks,
                            "page_from": page_buffer[0],
                            "page_to":   page_num,
                        }
                        if first_yield:
                            payload["page_count"] = total_pages
                            payload["file_hash"]  = file_hash
                            first_yield           = False

                        yield payload

                        # Reset buffers for next batch
                        pending_text = ""
                        page_buffer  = []

        except Exception as e:
            self.logger.error(f"Streaming extraction failed: {e}")
            yield {"status": "failed", "error": str(e)}

    # ------------------------------------------------------------------
    # Private: PDF / image extraction
    # ------------------------------------------------------------------

    def _extract_from_pdf(self, file_path: str) -> Dict:
        """
        Extract text and tables from a PDF.
        Logs progress every 50 pages for large documents.
        """
        content:    str       = ""
        tables:     List[Dict] = []
        page_count: int        = 0

        try:
            with pdfplumber.open(file_path) as pdf:
                page_count = len(pdf.pages)
                is_large   = page_count > 100

                for page_num, page in enumerate(pdf.pages, 1):
                    text     = page.extract_text() or ""
                    content += f"--- PAGE {page_num} ---\n{text}\n"

                    if is_large and page_num % 50 == 0:
                        self.logger.info(
                            f"  Extracted {page_num}/{page_count} pages …"
                        )

                    if EXTRACT_TABLES:
                        try:
                            extracted_tables = page.extract_tables()
                            if extracted_tables:
                                for table in extracted_tables:
                                    if not table or not table[0]:
                                        continue
                                    raw_headers = [
                                        str(h) if h is not None else ""
                                        for h in table[0]
                                    ]
                                    headers = self._deduplicate_columns(raw_headers)
                                    rows    = table[1:] if len(table) > 1 else []
                                    df      = pd.DataFrame(rows, columns=headers)
                                    tables.append({
                                        "page": page_num,
                                        "data": df.to_dict(orient="records")
                                    })
                        except Exception as e:
                            self.logger.warning(
                                f"Table extraction failed for page {page_num}: {e}"
                            )

            if OCR_ENABLED and len(content.strip()) < 500:
                self.logger.info("Applying OCR to scanned PDF")
                content = self._apply_ocr(file_path, content)

            return {
                "status":     "success",
                "content":    content,
                "page_count": page_count,
                "tables":     tables,
            }

        except Exception as e:
            return {"status": "failed", "error": f"PDF extraction error: {str(e)}"}

    def _extract_from_image(self, file_path: str) -> Dict:
        """Extract text from an image file using OCR."""
        try:
            image = Image.open(file_path)
            text  = pytesseract.image_to_string(image)
            return {"status": "success", "content": text, "page_count": 1}
        except Exception as e:
            return {"status": "failed", "error": f"Image extraction error: {str(e)}"}

    def _apply_ocr(self, file_path: str, existing_text: str) -> str:
        """Apply OCR to every page of a PDF and append to existing text."""
        ocr_text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    img      = page.to_image()
                    text     = pytesseract.image_to_string(img.original)
                    ocr_text += f"--- OCR PAGE {page_num} ---\n{text}\n"
        except Exception as e:
            self.logger.warning(f"OCR failed: {e}")

        return existing_text + ocr_text if ocr_text else existing_text

    # ------------------------------------------------------------------
    # Private: section detection
    # ------------------------------------------------------------------

    def _detect_sections(self, text: str) -> List[Dict]:
        """Detect standard tender sections by keyword matching."""
        sections:   List[Dict] = []
        text_lower: str        = text.lower()

        for section_type, patterns in self.SECTION_KEYWORDS.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    sections.append({
                        "type":       section_type,
                        "keyword":    pattern,
                        "confidence": 0.85,
                        "position":   match.start()
                    })
                    break   # Count each section type only once

        return sorted(sections, key=lambda x: x["position"])

    # ------------------------------------------------------------------
    # Private: basic regex metadata
    # ------------------------------------------------------------------

    def _extract_basic_metadata(self, text: str) -> Dict:
        """Extract basic metadata from raw text using regex heuristics."""
        metadata: Dict = {}

        tender_id_match = re.search(
            r"(?:tender\s+id|tender\s+number|ref\s+no)[:\s]+([A-Z0-9\-/]+)",
            text[:1000], re.IGNORECASE
        )
        if tender_id_match:
            metadata["tender_id"] = tender_id_match.group(1)

        budget_match = re.search(
            r"(?:budget|value|amount)[:\s]*(?:INR|USD|EUR)?\s*([0-9,\.]+)",
            text[:2000], re.IGNORECASE
        )
        if budget_match:
            try:
                metadata["budget"] = float(budget_match.group(1).replace(",", ""))
            except ValueError:
                pass

        deadline_match = re.search(
            r"(?:deadline|submission|due\s+date)[:\s]*"
            r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            text[:2000], re.IGNORECASE
        )
        if deadline_match:
            metadata["deadline"] = deadline_match.group(1)

        return metadata

    # ------------------------------------------------------------------
    # Private: chunking
    # ------------------------------------------------------------------

    def _create_chunks(self, text: str, sections: List[Dict]) -> List[Dict]:
        """
        Split text into overlapping fixed-size chunks for embedding.

        Strategy:
          - Split by page marker.
          - Slide a window of CHUNK_SIZE tokens with CHUNK_OVERLAP overlap.
          - Label each chunk with a section_type.

        Returns:
            List of {id, text, section_type, page, token_count, chunk_index}
        """
        chunks:   List[Dict] = []
        chunk_id: int        = 0
        pages                = text.split("--- PAGE")

        for page_num, page_text in enumerate(pages, 1):
            page_text = page_text.replace("---", "").strip()

            if len(page_text) < MIN_TEXT_LENGTH:
                continue

            section_type = self._section_for_page(sections, page_num, len(pages))
            tokens       = page_text.split()
            start        = 0

            while start < len(tokens):
                end          = min(start + CHUNK_SIZE, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text   = " ".join(chunk_tokens)

                if len(chunk_tokens) >= MIN_TEXT_LENGTH:
                    chunks.append({
                        "id":           chunk_id,
                        "text":         chunk_text,
                        "section_type": section_type,
                        "page":         page_num,
                        "token_count":  len(chunk_tokens),
                        "chunk_index":  chunk_id,
                    })
                    chunk_id += 1

                step  = max(CHUNK_SIZE - CHUNK_OVERLAP, 1)
                start += step

        return chunks

    def _section_for_page(
        self, sections: List[Dict], page_num: int, total_pages: int
    ) -> str:
        """Return the section type most likely associated with a given page."""
        if not sections:
            return "general"
        if total_pages <= 1:
            return sections[0]["type"]
        idx = min(
            int((page_num / total_pages) * len(sections)),
            len(sections) - 1
        )
        return sections[idx]["type"]

    # ------------------------------------------------------------------
    # Private: utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate_columns(headers: List[str]) -> List[str]:
        """
        Make DataFrame column names unique to prevent silent data loss.
        e.g. ["Name", "Name"] → ["Name", "Name_1"]
        """
        seen:   Dict[str, int] = {}
        result: List[str]      = []
        for col in headers:
            if col not in seen:
                seen[col] = 0
                result.append(col)
            else:
                seen[col] += 1
                result.append(f"{col}_{seen[col]}")
        return result

    @staticmethod
    def _compute_hash(file_path: str) -> str:
        """Compute SHA-256 hash of a file for deduplication."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()