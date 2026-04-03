"""
Multimodal PDF loader — production-ready, fast version.

Config flags (set at top of this file or via environment variables):
  USE_MULTIMODAL  : Master toggle. False = text-only, fastest path.
  USE_BLIP        : Toggle BLIP image captioning. Default False (slow, ~1 GB model).
  USE_CLOUD_VISION: Placeholder for future Groq/OpenAI vision API.

Content extracted per page:
  1. Raw text          — PyPDFLoader  (always active)
  2. Structured KV     — regex alias table (always active)
  3. Tables            — pdfplumber → markdown (when USE_MULTIMODAL=True)
  4. Image captions    — BLIP local (USE_BLIP=True) or placeholder text

Dependencies for multimodal path:
    pip install pdfplumber pymupdf tabulate
    # BLIP only (optional):
    pip install transformers Pillow torch
"""

import io
import os
import re

# ─── Config flags ─────────────────────────────────────────────────────────────
# Override these via environment variables for flexible deployment:
#   export USE_MULTIMODAL=true
#   export USE_BLIP=false

USE_MULTIMODAL: bool = os.getenv("USE_MULTIMODAL", "true").lower() == "true"
USE_BLIP: bool       = os.getenv("USE_BLIP",        "false").lower() == "true"
USE_CLOUD_VISION: bool = os.getenv("USE_CLOUD_VISION", "false").lower() == "true"

# ─── Lazy imports ─────────────────────────────────────────────────────────────
# Heavy dependencies are only imported when the relevant flag is True,
# so the module loads instantly when BLIP is disabled.

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')


# ─── Image captioning (pluggable) ─────────────────────────────────────────────

_blip_processor = None
_blip_model = None


def _caption_via_blip(img_bytes: bytes) -> str:
    """Local BLIP captioning — only called when USE_BLIP=True."""
    global _blip_processor, _blip_model
    try:
        if _blip_model is None:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            from PIL import Image

            print("[multimodal] Loading BLIP model (one-time, ~500 MB)...")
            _blip_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            _blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            print("[multimodal] BLIP ready.")

        from PIL import Image

        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        inputs = _blip_processor(image, return_tensors="pt")
        out = _blip_model.generate(**inputs, max_new_tokens=60)
        return _blip_processor.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        return f"[Image — caption failed: {e}]"


def _caption_via_cloud(img_bytes: bytes, page_num: int) -> str:
    """Stub for future cloud vision integration (Groq / OpenAI Vision).

    To enable:
      1. Set USE_CLOUD_VISION=true
      2. Implement the API call below using your preferred provider.
    """
    # TODO: implement Groq llava-v1.5-7b or OpenAI gpt-4o-mini vision call
    # Example skeleton:
    #   import base64, openai
    #   b64 = base64.b64encode(img_bytes).decode()
    #   resp = openai.chat.completions.create(
    #       model="gpt-4o-mini",
    #       messages=[{"role": "user", "content": [
    #           {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    #           {"type": "text", "text": "Describe this image in one sentence."}
    #       ]}]
    #   )
    #   return resp.choices[0].message.content
    return f"[Image on page {page_num + 1}: cloud vision not yet configured]"


def _get_image_caption(img_bytes: bytes, page_num: int) -> str:
    """Route image captioning based on active config flags."""
    if USE_BLIP:
        return _caption_via_blip(img_bytes)
    if USE_CLOUD_VISION:
        return _caption_via_cloud(img_bytes, page_num)
    # Default: fast placeholder — keeps image presence in the index
    # without any model overhead.
    return f"[Image present on page {page_num + 1}]"


# ─── Table extraction ─────────────────────────────────────────────────────────

def _extract_tables(pdf_path: str) -> dict:
    """Return {page_index: [table_markdown, ...]} using pdfplumber."""
    import pdfplumber
    from tabulate import tabulate

    results: dict[int, list[str]] = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                raw_tables = page.extract_tables()
                if not raw_tables:
                    continue
                markdowns = []
                for table in raw_tables:
                    if not table or not table[0]:
                        continue
                    header = table[0]
                    rows = table[1:]
                    try:
                        md = tabulate(rows, headers=header, tablefmt="pipe")
                    except Exception:
                        md = "\n".join(
                            [" | ".join(str(c or "") for c in row) for row in table]
                        )
                    markdowns.append(md)
                if markdowns:
                    results[i] = markdowns
    except Exception as e:
        print(f"[multimodal] Table extraction failed: {e}")
    return results


# ─── Image extraction ─────────────────────────────────────────────────────────

def _extract_image_captions(pdf_path: str) -> dict:
    """Return {page_index: [caption, ...]} using pymupdf."""
    import fitz  # pymupdf

    results: dict[int, list[str]] = {}
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            images = page.get_images(full=True)
            captions = []
            for img_info in images:
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    img_bytes = base_image["image"]
                    caption = _get_image_caption(img_bytes, page_num)
                    captions.append(caption)
                except Exception as e:
                    captions.append(
                        f"[Image on page {page_num + 1}: extraction failed — {e}]"
                    )
            if captions:
                results[page_num] = captions
        doc.close()
    except Exception as e:
        print(f"[multimodal] Image extraction failed: {e}")
    return results


# ─── Structured KV extraction ─────────────────────────────────────────────────

_KV_ALIASES: dict[str, str] = {
    # Authorship
    "by name":        "author",
    "prepared by":    "author",
    "submitted by":   "author",
    "authored by":    "author",
    "written by":     "author",
    "approved by":    "approved_by",
    "reviewed by":    "reviewed_by",
    "signed by":      "signed_by",
    # Dates
    "date":           "date",
    "created on":     "date",
    "submitted on":   "date",
    "prepared on":    "date",
    # Document identity
    "title":          "title",
    "subject":        "subject",
    "reference":      "reference",
    "ref":            "reference",
    "document no":    "document_no",
    "doc no":         "document_no",
    "version":        "version",
    "rev":            "version",
    # Organisation
    "company":        "organisation",
    "organisation":   "organisation",
    "organization":   "organisation",
    "department":     "department",
    "dept":           "department",
    "client":         "client",
    "project":        "project",
}

_KV_RE = re.compile(r'^([\w][\w /()&-]{1,40}?)\s*[:\-]\s*(.+)$', re.MULTILINE)


def _extract_kv_pairs(raw_text: str) -> str:
    """Detect Label: Value patterns and return normalised declarative sentences."""
    sentences: list[str] = []
    seen: set[str] = set()

    for match in _KV_RE.finditer(raw_text):
        label_raw = match.group(1).strip()
        value = match.group(2).strip()

        if not value or len(value) > 200:
            continue

        label_lower = label_raw.lower()
        canon = _KV_ALIASES.get(label_lower)

        if canon is None:
            for alias, key in _KV_ALIASES.items():
                if label_lower.startswith(alias) or alias.startswith(label_lower):
                    canon = key
                    break

        if canon is None:
            continue

        sentence = f"{canon} is {value}"
        if sentence not in seen:
            seen.add(sentence)
            sentences.append(sentence)

    if not sentences:
        return ""
    return "\n\n### Structured Fields:\n" + "\n".join(f"- {s}." for s in sentences)


def load_pdf_with_ocr(file_path: str, filename: str) -> list[Document]:
    import fitz
    import os
    print(f"[loader] Processing scanned PDF via PaddleOCR: {file_path}")
    try:
        doc = fitz.open(file_path)
        full_text = ""

        temp_img = f"temp_ocr_{os.path.basename(file_path)}.png"

        for i, page in enumerate(doc):
            pix = page.get_pixmap()
            pix.save(temp_img)

            result = ocr.ocr(temp_img)
            
            # PaddleOCR returns a list of results. Each result is a list of lines. 
            # line[1][0] contains the text.
            page_text = ""
            if result and result[0]:
                for line in result[0]:
                    page_text += line[1][0] + "\n"
            
            print(f"[DEBUG] Page {i+1} OCR length: {len(page_text)}")
            full_text += page_text

        # Clean up temp file
        if os.path.exists(temp_img):
            os.remove(temp_img)

        if not full_text.strip():
            print("[ERROR] OCR failed: no text extracted")
            return []

        # Return a single document representing the entire OCR'd PDF
        return [Document(
            page_content=full_text,
            metadata={"source": filename, "page": 0, "is_scanned": True}
        )]
    except Exception as e:
        print(f"[ERROR] OCR processing failed: {e}")
        return []


# ─── Public API ───────────────────────────────────────────────────────────────

def load_multimodal_pdf(pdf_path: str, filename: str) -> list:
    """Load a PDF and return enriched LangChain Documents (one per page).

    Metadata keys on every document:
      source    : filename
      page      : 0-based page index
      has_table : bool
      has_image : bool
    """
    print(f"[multimodal] Loading: {filename}  "
          f"(multimodal={USE_MULTIMODAL}, blip={USE_BLIP})")

    # 1. Base text (always)
    loader = PyPDFLoader(pdf_path)
    base_docs = loader.load()
    
    # 1.5 Scan check
    # Check if this is a scanned PDF with no extractable text
    total_stripped_text = sum(len(doc.page_content.strip()) for doc in base_docs)
    if total_stripped_text < 20:
        print("[INFO] Scanned PDF detected → switching to OCR")
        return load_pdf_with_ocr(pdf_path, filename)
        
    print("[INFO] Normal PDF detected")

    # 2. Tables + images — only when multimodal is enabled
    table_map: dict = {}
    image_map: dict = {}

    if USE_MULTIMODAL:
        table_map = _extract_tables(pdf_path)
        image_map = _extract_image_captions(pdf_path)

    # 3. Merge per page
    enriched_docs = []
    for doc in base_docs:
        page_num = doc.metadata.get("page", 0)
        content = doc.page_content.strip()

        # Structured KV (always — zero cost, pure regex)
        kv_block = _extract_kv_pairs(content)
        if kv_block:
            content += kv_block

        # Tables (markdown)
        for table_md in table_map.get(page_num, []):
            content += f"\n\n### Table (page {page_num + 1}):\n{table_md}"

        # Image captions / placeholders
        for caption in image_map.get(page_num, []):
            content += f"\n\n{caption}"

        enriched_docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": filename,
                    "page": page_num,
                    "has_table": page_num in table_map,
                    "has_image": page_num in image_map,
                },
            )
        )

    print(
        f"[multimodal] {len(enriched_docs)} pages enriched "
        f"({len(table_map)} with tables, {len(image_map)} with images)"
    )
    return enriched_docs
