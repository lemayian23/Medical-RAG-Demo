"""
Medical Named Entity Recognition (NER) Module
- Extracts: medicines, diseases, symptoms, dosages, procedures
- Uses: PubMedBERT or SpaCy with medical pipeline
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to load SpaCy medical model
try:
    import spacy
    # Load medical NLP model (install with: python -m spacy download en_core_sci_lg)
    try:
        nlp = spacy.load("en_core_sci_lg")
        logger.info("Loaded SpaCy biomedical model: en_core_sci_lg")
    except:
        # Fallback to general model
        nlp = spacy.load("en_core_web_sm")
        logger.info("Loaded SpaCy general model: en_core_web_sm")
except ImportError:
    nlp = None
    logger.warning("SpaCy not installed. Medical NER will use regex fallback.")


@dataclass
class MedicalEntity:
    """Medical entity extracted from text."""
    text: str
    type: str  # e.g., "DRUG", "DISEASE", "SYMPTOM"
    start: int
    end: int
    confidence: float = 1.0


class MedicalNER:
    """
    Medical Named Entity Recognition.
    Extracts medical entities using SpaCy + patterns.
    """

    # Common drug names (simplified for demo)
    DRUG_PATTERNS = [
        # Brand names
        r'\b(Paracetamol|Panadol|Acetaminophen|Ibuprofen|Aspirin|Omeprazole)\b',
        r'\b(Crocin|Dolo|Combiflam|Nicip|Diclofenac|Metformin|Glimepiride)\b',
        r'\b(Rosuvastatin|Atorvastatin|Simvastatin|Amlodipine|Enalapril|Losartan)\b',
        r'\b(Tamoxifen|Herceptin|Trastuzumab|Paclitaxel|Docetaxel|Carboplatin)\b',
        r'\b(Pertuzumab|Lapatinib|Neratinib|Kadcyla|T-DM1)\b',
        # Generic patterns
        r'\b([A-Z][a-z]+)(?:[a-z]*)?\s*(?:Tablet|Capsule|Injection|Syrup|Ointment)\b',
        r'\b([A-Z][a-z]+)\s*(\d+\s*(?:mg|g|ml|mcg|IU))?\b',
    ]

    # Disease/condition patterns
    DISEASE_PATTERNS = [
        r'\b(breast cancer|breast carcinoma|HER2-positive|HER2\\+)\b',
        r'\b(TNBC|triple negative|triple-negative)\b',
        r'\b(carcinoma|adenocarcinoma|sarcoma|melanoma)\b',
        r'\b(diabetes|hypertension|asthma|COPD|arthritis)\b',
        r'\b(cancer|tumor|neoplasm|malignancy)\b',
        r'\b(heart disease|stroke|myocardial infarction|MI)\b',
    ]

    # Symptom patterns
    SYMPTOM_PATTERNS = [
        r'\b(fever|headache|nausea|vomiting|dizziness|fatigue|pain)\b',
        r'\b(anxiety|depression|insomnia|confusion|delirium)\b',
        r'\b(rash|itching|swelling|redness|bleeding)\b',
        r'\b(shortness of breath|SOB|chest pain|palpitations)\b',
    ]

    # Dosage patterns
    DOSAGE_PATTERNS = [
        r'(\d+)\s*(mg|g|ml|mcg|IU|unit|tablet|capsule)',
        r'(\d+\.?\d*)\s*(mg/kg|g/kg|ml/kg)',
    ]

    def __init__(self):
        """Initialize NER."""
        self.use_spacy = nlp is not None
        if self.use_spacy:
            logger.info("Medical NER with SpaCy enabled")
        else:
            logger.warning("Medical NER using regex fallback (less accurate)")

    def extract_entities(self, text: str) -> List[MedicalEntity]:
        """
        Extract medical entities from text.
        """
        entities = []

        # Try SpaCy first
        if self.use_spacy:
            entities.extend(self._extract_spacy(text))

        # Always add regex-based extraction for specific patterns
        entities.extend(self._extract_regex(text))

        # Remove duplicates
        unique_entities = {}
        for e in entities:
            key = (e.text.lower(), e.type)
            if key not in unique_entities or e.confidence > unique_entities[key].confidence:
                unique_entities[key] = e

        return list(unique_entities.values())

    def _extract_spacy(self, text: str) -> List[MedicalEntity]:
        """Extract entities using SpaCy."""
        entities = []
        doc = nlp(text)

        for ent in doc.ents:
            # Map SpaCy entity types to medical types
            entity_type = self._map_spacy_type(ent.label_)
            if entity_type:
                entities.append(MedicalEntity(
                    text=ent.text,
                    type=entity_type,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.85,  # Average SpaCy confidence
                ))

        return entities

    def _map_spacy_type(self, spacy_type: str) -> Optional[str]:
        """Map SpaCy entity type to medical entity type."""
        mapping = {
            "DRUG": "DRUG",
            "GENE": "GENE",
            "DISEASE": "DISEASE",
            "SYMPTOM": "SYMPTOM",
            "DOSAGE": "DOSAGE",
            "MEDICATION": "DRUG",
            "PERSON": "PATIENT",
            "ORG": "DOCTOR",
        }
        return mapping.get(spacy_type)

    def _extract_regex(self, text: str) -> List[MedicalEntity]:
        """Extract entities using regex patterns."""
        entities = []

        # Extract drugs
        for pattern in self.DRUG_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(MedicalEntity(
                    text=match.group(1) if match.groups() else match.group(0),
                    type="DRUG",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.7,
                ))

        # Extract diseases
        for pattern in self.DISEASE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(MedicalEntity(
                    text=match.group(0),
                    type="DISEASE",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.7,
                ))

        # Extract symptoms
        for pattern in self.SYMPTOM_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(MedicalEntity(
                    text=match.group(0),
                    type="SYMPTOM",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.7,
                ))

        # Extract dosages
        for pattern in self.DOSAGE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(MedicalEntity(
                    text=match.group(0),
                    type="DOSAGE",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8,
                ))

        return entities

    def extract_structured_summary(self, text: str) -> Dict[str, List[str]]:
        """
        Extract structured summary from medical text.
        """
        entities = self.extract_entities(text)

        summary = {
            "drugs": [],
            "diseases": [],
            "symptoms": [],
            "dosages": [],
        }

        for e in entities:
            if e.type == "DRUG":
                summary["drugs"].append(e.text)
            elif e.type == "DISEASE":
                summary["diseases"].append(e.text)
            elif e.type == "SYMPTOM":
                summary["symptoms"].append(e.text)
            elif e.type == "DOSAGE":
                summary["dosages"].append(e.text)

        # Remove duplicates
        for key in summary:
            summary[key] = list(set(summary[key]))

        return summary


# Singleton instance
_ner_instance = None

def get_ner() -> MedicalNER:
    global _ner_instance
    if _ner_instance is None:
        _ner_instance = MedicalNER()
    return _ner_instance