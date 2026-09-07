"""
OCR Module - Medical Document Processing
Handles: PDFs, Images (JPG, PNG), Prescriptions, Scanned Documents
"""

import os
import re
import logging
from typing import Union, Dict, List, Optional
from pathlib import Path

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import pdfplumber
import PyPDF2
from pdf2image import convert_from_bytes, convert_from_path

from app.core.config import config

logger = logging.getLogger(__name__)


class MedicalOCR:
    """
    OCR for medical documents, prescriptions, and clinical notes.
    """

    def __init__(self):
        """Initialize OCR engine with medical-specific settings."""
        # Set Tesseract path if configured
        tesseract_path = os.getenv("TESSERACT_PATH")
        if tesseract_path and os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            logger.info(f"Tesseract path set: {tesseract_path}")

        # OCR config for medical text
        self.tesseract_config = os.getenv(
            "TESSERACT_CONFIG",
            '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789./-,:;()% '
        )
        self.language = os.getenv("OCR_LANGUAGE", "eng")
        self.dpi = int(os.getenv("OCR_DPI", 300))

        self.upload_dir = Path(os.getenv("UPLOAD_DIR", "./data/uploads"))
        self.processed_dir = Path(os.getenv("PROCESSED_DIR", "./data/processed"))

        # Create directories
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"OCR initialized with language: {self.language}")

    # ============================================================
    # MAIN PROCESSING METHODS
    # ============================================================

    def process_document(self, file_data: bytes, filename: str) -> Dict[str, Union[str, List, Dict]]:
        """
        Main entry point for processing any document.
        """
        ext = filename.split('.')[-1].lower()
        result = {
            "filename": filename,
            "file_type": ext,
            "text": "",
            "structured_data": {},
            "ocr_used": False,
            "success": False,
            "error": None,
        }

        try:
            if ext == 'pdf':
                result = self.process_pdf(file_data, filename)
            elif ext in ['jpg', 'jpeg', 'png', 'tiff', 'bmp']:
                result = self.process_image(file_data, filename)
            elif ext == 'txt':
                text = file_data.decode('utf-8')
                result["text"] = text
                result["success"] = True
            elif ext == 'docx':
                from docx import Document
                doc = Document(io.BytesIO(file_data))
                text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                result["text"] = text
                result["success"] = True
            else:
                result["error"] = f"Unsupported file format: {ext}"

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            result["error"] = str(e)

        return result

    # ============================================================
    # PDF PROCESSING
    # ============================================================

    def process_pdf(self, file_data: bytes, filename: str) -> Dict:
        """
        Process PDF: try digital extraction first, then OCR fallback.
        """
        result = {
            "filename": filename,
            "file_type": "pdf",
            "text": "",
            "structured_data": {},
            "ocr_used": False,
            "success": False,
            "error": None,
        }

        try:
            # Try digital text extraction first
            with pdfplumber.open(io.BytesIO(file_data)) as pdf:
                text_parts = []
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        text_parts.append(page_text)
                text = "\n\n".join(text_parts)

                if text.strip():
                    result["text"] = text
                    result["success"] = True
                    logger.info(f"PDF processed digitally: {filename}")
                    return result
        except Exception as e:
            logger.warning(f"Digital extraction failed for {filename}: {e}")

        # Fallback: OCR for scanned PDF
        try:
            logger.info(f"Using OCR on scanned PDF: {filename}")
            images = convert_from_bytes(file_data, dpi=self.dpi)
            text_parts = []
            for i, img in enumerate(images):
                page_text = self._ocr_image(img)
                text_parts.append(f"--- Page {i+1} ---\n{page_text}")

            result["text"] = "\n\n".join(text_parts)
            result["ocr_used"] = True
            result["success"] = True
            logger.info(f"PDF processed with OCR: {filename}")
            return result

        except Exception as e:
            result["error"] = f"OCR failed: {str(e)}"
            logger.error(f"OCR failed for {filename}: {e}")
            return result

    # ============================================================
    # IMAGE PROCESSING
    # ============================================================

    def process_image(self, file_data: bytes, filename: str) -> Dict:
        """
        Process an image with OCR and medical-specific extraction.
        """
        result = {
            "filename": filename,
            "file_type": filename.split('.')[-1].lower(),
            "text": "",
            "structured_data": {},
            "ocr_used": True,
            "success": False,
            "error": None,
        }

        try:
            image = Image.open(io.BytesIO(file_data))

            # Pre-process image for better OCR
            processed_image = self._preprocess_image(image)

            # OCR the image
            text = self._ocr_image(processed_image)

            if not text.strip():
                # Try with different pre-processing
                processed_image = self._preprocess_image_alt(image)
                text = self._ocr_image(processed_image)

            result["text"] = text
            result["success"] = True if text.strip() else False

            # Try to extract prescription data
            if text.strip():
                prescription_data = self.extract_prescription_data(text)
                if any(prescription_data.values()):
                    result["structured_data"] = prescription_data

            logger.info(f"Image processed with OCR: {filename} ({len(text)} chars)")
            return result

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Image processing failed for {filename}: {e}")
            return result

    # ============================================================
    # PRESCRIPTION EXTRACTION
    # ============================================================

    def extract_prescription_data(self, text: str) -> Dict:
        """
        Extract structured data from prescription text.
        """
        result = {
            "medicines": [],
            "dosages": [],
            "frequencies": [],
            "duration": None,
            "patient": None,
            "doctor": None,
            "date": None,
        }

        lines = text.split('\n')

        # Extract patient name
        patient_patterns = [
            r'(?:Patient|Name|Pt|Pt\.)\s*:?\s*([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s*(?:Age|DOB|Date)'
        ]
        for pattern in patient_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["patient"] = match.group(1).strip()
                break

        # Extract doctor name
        doctor_patterns = [
            r'(?:Dr|Doctor|Dr\.)\s*:?\s*([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'(?:Prescribed by|Physician)\s*:?\s*([A-Z][a-z]+\s+[A-Z][a-z]+)'
        ]
        for pattern in doctor_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["doctor"] = match.group(1).strip()
                break

        # Extract date
        date_pattern = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})'
        match = re.search(date_pattern, text)
        if match:
            result["date"] = match.group(1)

        # Extract medicines, dosages, frequencies
        # Simplified patterns for demo
        medicine_pattern = r'([A-Z][a-z]+)\s*(\d+\s*(?:mg|g|ml|mcg))?\s*([A-Z]{2,3})?'
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for medicine-like patterns
            match = re.search(r'([A-Z][a-z]+)\s*(\d+\s*(?:mg|g|ml|mcg))?', line)
            if match:
                med = match.group(1).strip()
                dosage = match.group(2).strip() if match.group(2) else ""
                if med and len(med) > 1:
                    result["medicines"].append(med)
                    if dosage:
                        result["dosages"].append(dosage)

            # Look for frequency
            freq_match = re.search(r'(OD|BD|TDS|QID|Q4H|Q6H|Q8H|Q12H)', line, re.IGNORECASE)
            if freq_match:
                result["frequencies"].append(freq_match.group(1).upper())

        # Remove duplicates
        result["medicines"] = list(set(result["medicines"]))
        result["dosages"] = list(set(result["dosages"]))
        result["frequencies"] = list(set(result["frequencies"]))

        return result

    # ============================================================
    # IMAGE PRE-PROCESSING
    # ============================================================

    def _preprocess_image(self, image: Image) -> Image:
        """Pre-process image for better OCR."""
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)

        # Sharpen
        image = image.filter(ImageFilter.SHARPEN)

        return image

    def _preprocess_image_alt(self, image: Image) -> Image:
        """Alternative pre-processing for hard-to-read images."""
        # Binarize (threshold)
        image = image.convert('L')
        image = image.point(lambda x: 255 if x > 160 else 0, '1')
        return image

    def _ocr_image(self, image: Image) -> str:
        """Perform OCR on an image."""
        return pytesseract.image_to_string(
            image,
            config=self.tesseract_config,
            lang=self.language
        )


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def save_uploaded_file(file_data: bytes, filename: str, directory: Path = None) -> Path:
    """Save uploaded file to disk."""
    if directory is None:
        directory = Path(os.getenv("UPLOAD_DIR", "./data/uploads"))

    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / filename

    with open(file_path, 'wb') as f:
        f.write(file_data)

    return file_path


# ============================================================
# SINGLETON INSTANCE
# ============================================================

_ocr_instance = None


def get_ocr() -> MedicalOCR:
    """Singleton pattern for OCR."""
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = MedicalOCR()
    return _ocr_instance