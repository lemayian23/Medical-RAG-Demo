"""
Medical Named Entity Recognition (NER) Module
Extracts: medicines, diseases, symptoms, dosages, procedures
Uses: Regex patterns + optional SpaCy medical pipeline
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ============================================================
# OPTIONAL SPACY IMPORT
# ============================================================

try:
    import spacy
    try:
        nlp = spacy.load("en_core_sci_lg")
        logger.info("Loaded SpaCy biomedical model: en_core_sci_lg")
    except:
        try:
            nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded SpaCy general model: en_core_web_sm")
        except:
            nlp = None
            logger.warning("No SpaCy model found. Using regex fallback only.")
except ImportError:
    nlp = None
    logger.warning("SpaCy not installed. Using regex fallback only.")


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class MedicalEntity:
    """Medical entity extracted from text."""
    text: str
    type: str  # DRUG, DISEASE, SYMPTOM, DOSAGE, PROCEDURE, PATIENT, DOCTOR
    start: int
    end: int
    confidence: float = 1.0


# ============================================================
# MEDICAL NER CLASS
# ============================================================

class MedicalNER:
    """
    Medical Named Entity Recognition.
    Extracts medical entities using regex patterns and optionally SpaCy.
    """

    # ---------- DRUG PATTERNS ----------
    DRUG_PATTERNS = [
        # Common drugs
        r'\b(Paracetamol|Panadol|Acetaminophen|Ibuprofen|Aspirin|Omeprazole)\b',
        r'\b(Crocin|Dolo|Combiflam|Nicip|Diclofenac|Metformin|Glimepiride)\b',
        r'\b(Rosuvastatin|Atorvastatin|Simvastatin|Amlodipine|Enalapril|Losartan)\b',
        r'\b(Tamoxifen|Herceptin|Trastuzumab|Paclitaxel|Docetaxel|Carboplatin)\b',
        r'\b(Pertuzumab|Lapatinib|Neratinib|Kadcyla|T-DM1|Taxol)\b',
        r'\b(Amoxicillin|Azithromycin|Ciprofloxacin|Ceftriaxone|Doxycycline)\b',
        r'\b(Insulin|Metformin|Glibenclamide|Sitagliptin|Empagliflozin)\b',
        r'\b(Morphine|Tramadol|Codeine|Fentanyl|Oxycodone)\b',
        r'\b(Warfarin|Heparin|Clopidogrel|Rivaroxaban|Apixaban)\b',
        # Generic patterns
        r'\b([A-Z][a-z]+)(?:[a-z]*)?\s*(?:Tablet|Capsule|Injection|Syrup|Ointment|Suspension)\b',
        r'\b([A-Z][a-z]+)\s*(\d+\s*(?:mg|g|ml|mcg|IU))?\b',
    ]

    # ---------- DISEASE PATTERNS ----------
    DISEASE_PATTERNS = [
        r'\b(breast cancer|breast carcinoma|HER2-positive|HER2\\+)\b',
        r'\b(TNBC|triple negative|triple-negative)\b',
        r'\b(carcinoma|adenocarcinoma|sarcoma|melanoma|lymphoma|leukemia)\b',
        r'\b(diabetes|diabetes mellitus|type 1 diabetes|type 2 diabetes)\b',
        r'\b(hypertension|high blood pressure|asthma|COPD|arthritis)\b',
        r'\b(cancer|tumor|neoplasm|malignancy|metastasis)\b',
        r'\b(heart disease|stroke|myocardial infarction|MI|coronary artery disease)\b',
        r'\b(COVID-19|coronavirus|influenza|pneumonia|tuberculosis|TB)\b',
        r'\b(malaria|typhoid|cholera|HIV|AIDS|hepatitis)\b',
        r'\b(depression|anxiety|bipolar disorder|schizophrenia|PTSD)\b',
    ]

    # ---------- SYMPTOM PATTERNS ----------
    SYMPTOM_PATTERNS = [
        r'\b(fever|headache|nausea|vomiting|dizziness|fatigue|pain)\b',
        r'\b(anxiety|depression|insomnia|confusion|delirium|lethargy)\b',
        r'\b(rash|itching|swelling|redness|bleeding|bruising)\b',
        r'\b(shortness of breath|SOB|chest pain|palpitations|tachycardia)\b',
        r'\b(cough|sneezing|runny nose|sore throat|congestion)\b',
        r'\b(diarrhea|constipation|abdominal pain|bloating|indigestion)\b',
        r'\b(weight loss|weight gain|loss of appetite|night sweats)\b',
        r'\b(numbness|tingling|weakness|paralysis|tremor)\b',
        r'\b(blurred vision|double vision|hearing loss|tinnitus)\b',
    ]

    # ---------- DOSAGE PATTERNS ----------
    DOSAGE_PATTERNS = [
        r'(\d+)\s*(mg|g|ml|mcg|IU|unit|tablet|capsule)',
        r'(\d+\.?\d*)\s*(mg/kg|g/kg|ml/kg)',
        r'(\d+)\s*(mg|g|ml|mcg)\s*(?:once|twice|three times|four times)',
        r'(OD|BD|TDS|QID|Q4H|Q6H|Q8H|Q12H|PRN|STAT)',
    ]

    # ---------- FREQUENCY PATTERNS ----------
    FREQUENCY_PATTERNS = [
        r'\b(once daily|once a day|OD|daily)\b',
        r'\b(twice daily|twice a day|BD|BID)\b',
        r'\b(three times daily|three times a day|TDS|TID)\b',
        r'\b(four times daily|four times a day|QID)\b',
        r'\b(every \d+ hours|Q\d+H)\b',
        r'\b(as needed|PRN|when required)\b',
    ]

    def __init__(self):
        """Initialize NER."""
        self.use_spacy = nlp is not None
        if self.use_spacy:
            logger.info("Medical NER with SpaCy enabled")
        else:
            logger.warning("Medical NER using regex fallback (less accurate)")

    # ============================================================
    # MAIN EXTRACTION METHODS
    # ============================================================

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

        # Remove duplicates (keep highest confidence)
        unique_entities = {}
        for e in entities:
            key = (e.text.lower().strip(), e.type)
            if key not in unique_entities or e.confidence > unique_entities[key].confidence:
                unique_entities[key] = e

        return list(unique_entities.values())

    def _extract_spacy(self, text: str) -> List[MedicalEntity]:
        """Extract entities using SpaCy."""
        entities = []
        try:
            doc = nlp(text)
            for ent in doc.ents:
                entity_type = self._map_spacy_type(ent.label_)
                if entity_type:
                    entities.append(MedicalEntity(
                        text=ent.text,
                        type=entity_type,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=0.85,
                    ))
        except Exception as e:
            logger.warning(f"SpaCy extraction failed: {e}")
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
            "PROCEDURE": "PROCEDURE",
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

        # Extract frequencies
        for pattern in self.FREQUENCY_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(MedicalEntity(
                    text=match.group(0),
                    type="FREQUENCY",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.7,
                ))

        return entities

    # ============================================================
    # STRUCTURED EXTRACTION
    # ============================================================

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
            "frequencies": [],
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
            elif e.type == "FREQUENCY":
                summary["frequencies"].append(e.text)

        # Remove duplicates
        for key in summary:
            summary[key] = list(set(summary[key]))

        return summary

    # ============================================================
    # PRESCRIPTION EXTRACTION
    # ============================================================

    def extract_prescription_data(self, text: str) -> Dict[str, Any]:
        """
        Extract structured prescription data.
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

        # Extract duration
        duration_pattern = r'(?:for|duration)\s*(\d+)\s*(days|weeks|months)'
        match = re.search(duration_pattern, text, re.IGNORECASE)
        if match:
            result["duration"] = f"{match.group(1)} {match.group(2)}"

        # Extract medicines, dosages, frequencies
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Medicine pattern
            med_match = re.search(r'([A-Z][a-z]+)\s*(\d+\s*(?:mg|g|ml|mcg))?', line)
            if med_match:
                med = med_match.group(1).strip()
                dosage = med_match.group(2).strip() if med_match.group(2) else ""
                if med and len(med) > 1 and med not in ["The", "This", "That"]:
                    result["medicines"].append(med)
                    if dosage:
                        result["dosages"].append(dosage)

            # Frequency pattern
            freq_match = re.search(r'(OD|BD|TDS|QID|Q4H|Q6H|Q8H|Q12H|PRN|STAT|daily|twice|three times)', line, re.IGNORECASE)
            if freq_match:
                result["frequencies"].append(freq_match.group(1).upper())

        # Remove duplicates
        result["medicines"] = list(set(result["medicines"]))
        result["dosages"] = list(set(result["dosages"]))
        result["frequencies"] = list(set(result["frequencies"]))

        return result


# ============================================================
# SINGLETON INSTANCE
# ============================================================

_ner_instance = None


def get_ner() -> MedicalNER:
    """Singleton pattern for NER."""
    global _ner_instance
    if _ner_instance is None:
        _ner_instance = MedicalNER()
    return _ner_instance