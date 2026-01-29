"""
Legal Document Parser.

Uses a carefully crafted extraction prompt for Turkish legal texts.
Critical features:
- Multi-article splitting (HUMK 179/3 ve 75/2 → 2 statutes)
- Abbreviation resolution (HMK → 6100)
- Treaty detection (Türk-Alman SGS → TR-DE-SGS)
- Original text preservation for claim_summary

Usage:
    python -m src.parser --input data/raw --output data/parsed.json
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Load .env for API keys
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =============================================================================
# KNOWN LAWS & ABBREVIATIONS (Critical for statute normalization)
# =============================================================================

KNOWN_LAWS = {
    # İş Hukuku
    "4857": "İş Kanunu",
    "1475": "İş Kanunu (1475)",
    "6356": "Sendikalar ve Toplu İş Sözleşmesi Kanunu",
    "854": "Deniz İş Kanunu",
    "5521": "İş Mahkemeleri Kanunu",
    # Sosyal Güvenlik
    "5510": "Sosyal Sigortalar ve Genel Sağlık Sigortası Kanunu",
    "506": "Sosyal Sigortalar Kanunu",
    "1479": "Bağ-Kur Kanunu",
    "5434": "Emekli Sandığı Kanunu",
    "3201": "Yurt Dışı Borçlanma Kanunu",
    # Usul Hukuku
    "6100": "Hukuk Muhakemeleri Kanunu",
    "1086": "Hukuk Usulü Muhakemeleri Kanunu",
    "5271": "Ceza Muhakemesi Kanunu",
    # Borçlar/Medeni
    "6098": "Türk Borçlar Kanunu",
    "818": "Borçlar Kanunu",
    "4721": "Türk Medeni Kanunu",
    # Ticaret
    "6102": "Türk Ticaret Kanunu",
}

# Abbreviation → Law Number (Critical for normalization!)
ABBREVIATION_MAP = {
    "HUMK": "1086",
    "HMK": "6100",
    "BK": "818",
    "TBK": "6098",
    "İşK": "4857",
    "İş K.": "4857",
    "SGK": "5510",
    "SSGSSK": "5510",
    "SSK": "506",
    "TTK": "6102",
    "TMK": "4721",
    "CMK": "5271",
}

# Treaty patterns
TREATY_PATTERNS = [
    (r"Türk[- ]Alman", "TR-DE-SGS"),
    (r"Türk[- ]Avusturya", "TR-AT-SGS"),
    (r"Türk[- ]Hollanda", "TR-NL-SGS"),
]

# Known case types (semi-dynamic list for LLM)
KNOWN_CASE_TYPES = {
    "ISE_IADE": "Reinstatement",
    "ALACAK_TAZMINAT": "Compensation Claims",
    "HIZMET_TESPITI": "Service Determination",
    "IS_KAZASI": "Work Accident",
    "GOREV_UYUSMAZLIGI": "Jurisdiction Dispute",
}


# =============================================================================
# EXTRACTION PROMPT (Single unified prompt with dynamic output format)
# =============================================================================

def _build_known_statutes_section() -> str:
    """Build the known statutes section for the prompt from KNOWN_LAWS dictionary."""
    lines = []
    for law_no, law_name in KNOWN_LAWS.items():
        lines.append(f"- `{law_no}`: {law_name}")
    return "\n".join(lines)


def _build_known_case_types_section() -> str:
    """Build the known case types section for the prompt from KNOWN_CASE_TYPES dictionary."""
    lines = []
    for case_type, description in KNOWN_CASE_TYPES.items():
        lines.append(f"* `{case_type}`: {description}")
    return "\n".join(lines)


PROMPT_LAWYER_STYLE = """
Sen, Yargıtay kararlarını inceleyen ve bunları "Hukuksal Tahmin Modeli" (Legal AI) eğitimi için hazırlayan uzman bir Veri Mühendisisin.

GÖREVİN:
Karar metnini analiz et ve davanın en başına dönerek, **Davacı Avukatının Dava Dilekçesinde yazdığı "Olay Örgüsü" ve "Talepleri"** tersine mühendislikle (reverse engineering) yeniden oluştur.

### 🎯 1. ADIM: BİLGİ MADENCİLİĞİ (Dedektif Modu)
Metnin içinden şu parçaları topla (özellikle "Somut olayda", "Dosya içeriğine göre" kısımlarından):
* **Kim?** (Meslek: Fayans ustası, Kamyon şoförü, Gemi adamı)
* **Nerede?** (Şantiye, Gemi, Yurt dışı/Sudan, Zeytinlik)
* **Ne Oldu?** (İş kazası, Haksız fesih, Maaş ödenmedi, Sigorta yapılmadı)
* **Ne İstiyor?** (Kıdem tazminatı, Hizmet tespiti, İşe iade)

### 📋 2. ADIM: DAVA TÜRÜ (case_type_enum) - YARI DİNAMİK
Davanın konusunu en iyi anlatan etiketi belirle.

**ÖNCELİKLİ LİSTE (Mümkünse bunlardan seç):**
{known_case_types}

⚠️ **İstisna:** Eğer dava bu kategorilere HİÇ UYMUYORSA (Örn: "Sendikal Yetki", "Rekabet Yasağı", "Basın İş Kanunu"), o zaman içeriği en iyi özetleyen **KISA_VE_BÜYÜK_HARFLİ** yeni bir etiket üret (Örn: `SENDIKAL_YETKI`).

### ⚖️ 3. ADIM: KARAR SONUCU (outcome_enum)
Metnin "HÜKÜM" kısmına bak.
* `ONAMA`: Onama, Düzelterek Onama, Esastan Red.
* `BOZMA`: Bozma, Kararın Kaldırılması.
* `GOREVSIZLIK`: Görevsizlik, Yetkisizlik, Gönderme, Merci Tayini.
* `GERI_CEVIRME`: Dosyanın geri çevrilmesi.
* `null`: Eğer sonuç metinde yazmıyorsa veya belirsizse (Zorlama yapma, null bırak).

### ✍️ 4. ADIM: YENİDEN YAZMA (Avukat Üslubu Modu)
Topladığın bilgileri birleştirerek `plaintiff_arguments` alanını oluştur. Ancak bunu yaparken rastgele cümleler kurma. Aşağıdaki **"Dilekçe Şablonu"**na sadık kal:

**Kullanılacak Şablon (Syntax):**
> "Davacı vekili; müvekkilinin **[Tarih/Yer/Meslek]** olarak çalıştığını, **[Olay: Kaza/Fesih/Mobbing]** gerekçesiyle mağdur olduğunu iddia ederek **[Talep Edilen Haklar]** istemiştir."

**Yasaklı İfadeler (Bunları Kullanma):**
❌ "Mahkemece tespit edilmiştir..." (Bu hakimin ağzı)
❌ "Dosya incelendiğinde..."
❌ "Davalının itirazı üzerine..."

### 📜 5. ADIM: KANUN VE ETİKETLEME

**BİLİNEN KANUN LİSTESİ (Öncelikli Kullan!):**
{known_statutes}

**ÖNEMLİ:** Eğer metinde geçen kanun bu listede varsa, mutlaka listedeki ID ve ismi kullan. Listede yoksa yeni bir kanun ekleyebilirsin.

* **statutes:** Kanun maddelerini `LAW-ARTICLE` formatında çıkar (Örn: `4857-25`, `6100-22`).

{output_format}

## METİN:
{text}
"""

# Backward compatibility alias
PROMPT_BASE = PROMPT_LAWYER_STYLE

# Output format for JSON (Gemini)
JSON_OUTPUT_FORMAT = """### ÇIKTI FORMATI (JSON)
Sadece JSON ver:

```json
{{
  "plaintiff_arguments": "Davacı vekili; müvekkilinin davalı şirkete ait Türk bayraklı gemide 2. Kaptan olarak çalıştığını, sefer primlerinin ödenmediğini ve iş akdinin haksız feshedildiğini ileri sürerek Deniz İş Kanunu uyarınca kıdem tazminatı ve ücret alacaklarını talep etmiştir.",
  "case_type_enum": "ALACAK_TAZMINAT",
  "outcome_enum": "GOREVSIZLIK",
  "chamber": "9. Hukuk Dairesi",
  "statute_ids": ["854-1", "6100-22"]
}}
```"""

# Output format for structured text (OpenAI)
TEXT_OUTPUT_FORMAT = """## ÇIKTI FORMATI
Yanıtını AŞAĞIDAKİ FORMATTA ver (her satır ayrı):

PLAINTIFF: [davacı argümanları - orijinal metinden]
CASE_TYPE: [ISE_IADE, ALACAK_TAZMINAT, HIZMET_TESPITI, IS_KAZASI, GOREV_UYUSMAZLIGI veya yeni etiket]
OUTCOME: [ONAMA, BOZMA, GOREVSIZLIK, GERI_CEVIRME veya null]
STATUTES: [virgülle ayrılmış kanun maddeleri: 4857-25, 6100-22, 1086-179/3]
CHAMBER: [daire adı örn: 9. Hukuk Dairesi]"""

# Build dynamic prompt with known statutes and case types
def build_extraction_prompt(output_format: str = JSON_OUTPUT_FORMAT) -> str:
    """Build the extraction prompt with dynamically injected known statutes and case types."""
    known_statutes = _build_known_statutes_section()
    known_case_types = _build_known_case_types_section()
    return PROMPT_LAWYER_STYLE.format(
        known_statutes=known_statutes,
        known_case_types=known_case_types,
        output_format=output_format,
        text="{text}"
    )

# For backwards compatibility
EXTRACTION_PROMPT = build_extraction_prompt(JSON_OUTPUT_FORMAT)





def load_text_from_file(file_path: Path) -> str:
    """Load text from raw file, handling nested JSON structures."""
    
    if file_path.suffix == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not isinstance(data, dict):
        return str(data)
    
    for key in ["raw_text", "raw", "data", "text"]:
        if key not in data or not data[key]:
            continue
        
        candidate = data[key]
        if not isinstance(candidate, str):
            continue
        
        if candidate.strip().startswith("{"):
            match = re.search(r'"data"\s*:\s*"((?:[^"\\]|\\.)*)"', candidate)
            if match:
                text = match.group(1)
                text = text.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
                return text
        
        return candidate
    
    return ""


# Error patterns in raw API responses that should be skipped
ERROR_PATTERNS = [
    "ADALET_RUNTIME_EXCEPTION",
    "DisplayCaptcha",
    "RUNTIME_EXCEPTION",
    '"data":null',
]


def is_error_document(text: str, file_path: Path = None) -> bool:
    """Check if the raw text is an error response and should be skipped."""
    if not text:
        return True
    
    # Check for error patterns
    for pattern in ERROR_PATTERNS:
        if pattern in text:
            return True
    
    return False


def clean_text(text: str) -> str:
    """Basic text cleaning."""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# =============================================================================
# STATUTE NORMALIZATION (Critical logic from the prompt!)
# =============================================================================

def normalize_statute(law_no: str, article: str) -> Optional[str]:
    """
    Normalize a statute reference to canonical ID.
    
    Handles:
    - Abbreviation resolution (HMK → 6100)
    - Roman numeral conversion (25/II → 25/2)
    - Treaty IDs (TR-DE-SGS)
    - Canonical ID building (4857-25)
    """
    law_no = str(law_no).strip()
    
    # Check if it's a treaty ID (contains letters and dashes like TR-DE-SGS)
    if re.match(r'^[A-Z]{2}-[A-Z]{2}', law_no.upper()):
        # It's a treaty, keep as-is
        if article:
            return f"{law_no}-{article}"
        return law_no
    
    # Resolve abbreviation
    law_no_upper = law_no.upper()
    for abbrev, num in ABBREVIATION_MAP.items():
        if abbrev.upper() == law_no_upper or abbrev.upper() in law_no_upper:
            law_no = num
            break
    else:
        # Not an abbreviation, extract digits
        law_no = re.sub(r'[^\d]', '', law_no)
    
    if not law_no:
        return None
    
    # Handle article
    if article:
        article = str(article).strip()
        # Roman numeral conversion (handle both /II and -II patterns)
        roman_map = [("III", "3"), ("II", "2"), ("IV", "4"), ("I", "1"), ("V", "5")]  # Order matters!
        for roman, arabic in roman_map:
            article = re.sub(rf'/({roman})(?![IVX])', f'/{arabic}', article)  # Avoid partial match
        
        return f"{law_no}-{article}"
    
    return law_no


def extract_statutes_regex(text: str) -> List[Dict]:
    """
    Extract statutes using regex patterns.
    
    Handles:
    - "4857 sayılı Kanun'un 25. maddesi"
    - "HMK'nın 21 ve 22. maddeleri" (multi-article!)
    - "Türk-Alman Sosyal Güvenlik Sözleşmesi 29/4"
    """
    statutes = []
    
    # Pattern 1: "XXXX sayılı Kanun'un YY. maddesi" or "XXXX sayılı Kanun'un YY/II maddesi"
    for match in re.finditer(r'(\d{3,5})\s*sayılı[^,\.]*?(\d+(?:[/\-][IVX\d]+)?(?:[/\-][a-z])?)', text, re.IGNORECASE):
        article = match.group(2)
        if re.search(r'madde', text[match.end():match.end()+20], re.IGNORECASE):
            statutes.append({"law_no": match.group(1), "article": article})
    
    # Pattern 2: Abbreviations with article numbers
    # Handles: "HUMK'nun 179/3 maddesi", "HMK'nın 21 ve 22. maddeleri", "HMK 21. madde"
    abbrev_pattern = r"(HUMK|HMK|BK|TBK|İşK|İş K\.|SGK|SSK|TTK|TMK|CMK)[''`]?(?:'?nun|'?nın|'nın)?[^\d]{0,15}?(\d+(?:/[IVX\d]+)?(?:\s*ve\s*\d+(?:/[IVX\d]+)?)*)"
    
    for match in re.finditer(abbrev_pattern, text, re.IGNORECASE):
        # Check if "madde" follows within 30 chars
        remaining = text[match.end():match.end()+30]
        if not re.search(r'madde', remaining, re.IGNORECASE):
            continue
            
        abbrev = match.group(1).upper().replace(" ", "").replace(".", "")
        articles_str = match.group(2)
        
        # Resolve abbreviation
        law_no = ABBREVIATION_MAP.get(abbrev, ABBREVIATION_MAP.get(abbrev.replace("İŞK", "İşK"), abbrev))
        
        # Handle "21 ve 22" or "179/3" or "21 ve 179/3"
        article_parts = re.split(r'\s+ve\s+', articles_str)
        for art in article_parts:
            art = art.strip()
            if art:
                statutes.append({"law_no": law_no, "article": art})
    
    # Pattern 3: Treaties (Türk-Alman, Türk-Avusturya, etc.)
    for pattern, treaty_id in TREATY_PATTERNS:
        # Match: "Türk-Alman Sosyal Güvenlik Sözleşmesinin 29/4 maddesi"
        treaty_pattern = pattern + r'[^\d]{0,50}?(\d+(?:/\d+)?)[^\w]*madde'
        treaty_match = re.search(treaty_pattern, text, re.IGNORECASE)
        if treaty_match:
            statutes.append({"law_no": treaty_id, "article": treaty_match.group(1)})
    
    return statutes


# =============================================================================
# METADATA EXTRACTION
# =============================================================================

def extract_metadata(text: str) -> Dict:
    """Extract chamber, year, and outcome from text."""
    
    result = {"chamber": None, "year": None, "outcome": None}
    
    # Chamber
    chamber_match = re.search(r'(\d{1,2})\.\s*Hukuk\s+Dairesi', text, re.IGNORECASE)
    if chamber_match:
        result["chamber"] = f"{chamber_match.group(1)}. Hukuk Dairesi"
    elif "Hukuk Genel Kurulu" in text:
        result["chamber"] = "Hukuk Genel Kurulu"
    
    # Year
    year_match = re.search(r'(\d{4})/\d+\s*[EK]', text)
    if year_match:
        result["year"] = int(year_match.group(1))
    
    # Outcome
    upper_text = text.upper()
    if "BOZULMASINA" in upper_text or re.search(r'B\s*O\s*Z\s*U\s*L\s*M\s*A\s*S\s*I\s*N\s*A', upper_text):
        result["outcome"] = "BOZMA"
    elif "ONANMASINA" in upper_text or re.search(r'O\s*N\s*A\s*N\s*M\s*A\s*S\s*I\s*N\s*A', upper_text):
        result["outcome"] = "ONAMA"
    elif "DÜZELTEREK" in upper_text:
        result["outcome"] = "KISMI"
    elif "YETKİSİZLİK" in upper_text or "GÖREVSİZLİK" in upper_text:
        result["outcome"] = "USUL"
    
    return result


# =============================================================================
# PARSING
# =============================================================================

def parse_with_gemini(text: str, model) -> Dict[str, Any]:
    """Parse using Gemini API."""
    
    prompt = EXTRACTION_PROMPT.format(text=text[:6000])
    
    try:
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Gemini error: {e}")
        return {}

def parse_openai_response(response_text: str) -> Dict[str, Any]:
    """Parse structured text response from OpenAI using regex."""
    result = {
        "plaintiff_arguments": "",
        "statute_ids": [],
        "case_type_enum": None,
        "outcome": None,
        "chamber": None,
    }
    
    # Extract PLAINTIFF (stops at CASE_TYPE or OUTCOME or STATUTES)
    match = re.search(r'PLAINTIFF:\s*(.+?)(?=\nCASE_TYPE:|\nOUTCOME:|\nSTATUTES:|$)', response_text, re.DOTALL | re.IGNORECASE)
    if match:
        result["plaintiff_arguments"] = match.group(1).strip()
    
    # Extract CASE_TYPE
    match = re.search(r'CASE_TYPE:\s*(\S+)', response_text, re.IGNORECASE)
    if match:
        result["case_type_enum"] = match.group(1).strip().upper()
    
    # Extract OUTCOME
    match = re.search(r'OUTCOME:\s*(\S+)', response_text, re.IGNORECASE)
    if match:
        outcome = match.group(1).strip().upper()
        if outcome in ["ONAMA", "BOZMA", "GOREVSIZLIK", "GERI_CEVIRME"]:
            result["outcome"] = outcome
        elif "BOZMA" in outcome:
            result["outcome"] = "BOZMA"
        elif "ONAMA" in outcome:
            result["outcome"] = "ONAMA"
        elif "GOREV" in outcome or "YETKI" in outcome:
            result["outcome"] = "GOREVSIZLIK"
    
    # Extract STATUTES as simple IDs
    match = re.search(r'STATUTES:\s*(.+?)(?=\nCHAMBER:|$)', response_text, re.DOTALL | re.IGNORECASE)
    if match:
        statutes_str = match.group(1).strip()
        for s in re.split(r'[,\s]+', statutes_str):
            s = s.strip()
            if s and len(s) > 2 and '-' in s:
                result["statute_ids"].append(s)
    
    # Extract CHAMBER
    match = re.search(r'CHAMBER:\s*(.+?)(?=\n|$)', response_text, re.IGNORECASE)
    if match:
        result["chamber"] = match.group(1).strip()
    
    return result


def parse_with_openai(text: str, client) -> Dict[str, Any]:
    """Parse using OpenAI GPT-4o-mini with structured text output and retry logic."""
    
    base_prompt = build_extraction_prompt(TEXT_OUTPUT_FORMAT)
    prompt = base_prompt.format(text=text[:8000])
    
    max_retries = 5
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Sen bir Türk hukuk uzmanısın. Belirtilen formatta yanıt ver."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=5000,
            )
            
            response_text = response.choices[0].message.content.strip()
            return parse_openai_response(response_text)
            
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limit" in error_str.lower():
                # Rate limit - wait and retry with exponential backoff
                wait_time = (2 ** attempt) + 0.5  # 1.5, 2.5, 4.5, 8.5, 16.5 seconds
                time.sleep(wait_time)
                continue
            else:
                print(f"OpenAI error: {e}")
                return {}
    
    print(f"OpenAI error: Rate limit exhausted after {max_retries} retries")
    return {}



def parse_with_regex(text: str) -> Dict[str, Any]:
    """Regex-based extraction as fallback."""
    
    result = {
        "claim_summary": "",
        "plaintiff_arguments": "",
        "statutes": extract_statutes_regex(text),
        **extract_metadata(text)
    }
    
    # Extract plaintiff arguments section (detailed)
    # Look for "Davacı vekili, dava dilekçesinde..." pattern
    plaintiff_match = re.search(
        r'Davacı\s+vekili[^,]*,\s*(?:dava\s+dilekçesinde\s*)?(.*?)(?=\n\s*Davalı|\n\s*Mahkeme|\n\s*GEREKÇE|\n\s*İddia\s+ve\s+savunma)',
        text, re.DOTALL | re.IGNORECASE
    )
    if plaintiff_match:
        args = plaintiff_match.group(1).strip()
        # Remove any court decision text that might have crept in
        args = re.split(r'ONANMASINA|BOZULMASINA|Hüküm', args, flags=re.IGNORECASE)[0]
        result["plaintiff_arguments"] = args.strip()[:2000]  # Keep up to 2000 chars
        
        # Create short claim summary from plaintiff arguments
        result["claim_summary"] = args.strip()[:300]
    else:
        # Fallback: look for simpler "Davacı," pattern
        claim_match = re.search(
            r'Davacı[^,]*,\s*(.*?)(?=\n\s*Davalı|\n\s*Mahkeme|\n\s*GEREKÇE)',
            text, re.DOTALL | re.IGNORECASE
        )
        if claim_match:
            claim = claim_match.group(1).strip()
            claim = re.split(r'ONANMASINA|BOZULMASINA', claim, flags=re.IGNORECASE)[0]
            result["claim_summary"] = claim.strip()[:500]
    
    return result


def process_result(result: Dict, case_id: str, text: str) -> Dict:
    """Process extraction result to final format."""
    
    # Get metadata from regex as fallback
    meta = extract_metadata(text)
    
    # Handle statute_ids - can be list of strings or list of dicts
    statute_ids = []
    seen = set()
    
    # New format: statute_ids is already a list of strings like ["4857-25", "6100-22"]
    if result.get("statute_ids"):
        for s in result["statute_ids"]:
            if isinstance(s, str) and s not in seen:
                statute_ids.append(s)
                seen.add(s)
    # Old format: statutes is a list of dicts with law_no and article
    elif result.get("statutes"):
        for s in result["statutes"]:
            sid = normalize_statute(s.get("law_no", ""), s.get("article", ""))
            if sid and sid not in seen:
                statute_ids.append(sid)
                seen.add(sid)
    
    # Map outcome_enum to outcome for backward compatibility with graph builder
    outcome = result.get("outcome_enum") or result.get("outcome") or meta["outcome"]
    
    return {
        "id": case_id,
        "plaintiff_arguments": result.get("plaintiff_arguments", ""),
        "case_type_enum": result.get("case_type_enum"),
        "outcome": outcome,
        "chamber": result.get("chamber") or meta["chamber"],
        "statute_ids": statute_ids,
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def parse_files(
    input_dir: str,
    output_file: str,
    use_gemini: bool = False,
    use_openai: bool = False,
    limit: int = None,
    fail_fast: bool = True,
    workers: int = 1,
):
    """Parse all files in input directory."""
    
    input_path = Path(input_dir)
    output_path = Path(output_file)
    
    files = list(input_path.glob("*.json")) + list(input_path.glob("*.txt"))
    files = [f for f in files if not f.name.startswith("_")]
    
    if limit:
        files = files[:limit]
    
    print(f"Found {len(files)} files")
    
    # Setup LLM
    gemini_model = None
    openai_client = None
    
    if use_openai and HAS_OPENAI:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            openai_client = OpenAI(api_key=api_key)
            print(f"Using OpenAI GPT-4o-mini with {workers} workers")
        else:
            print("No OPENAI_API_KEY found")
            use_openai = False
    
    if use_gemini and HAS_GEMINI and not openai_client:
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel("gemini-1.5-flash")
            print("Using Gemini API (sequential due to rate limits)")
            workers = 1  # Gemini has strict rate limits
        else:
            print("No GEMINI_API_KEY found")
            use_gemini = False
    
    if not openai_client and not gemini_model:
        print("Using regex-only parsing (no LLM)")
    
    results = []
    errors = 0
    
    def process_single_file(file_path):
        """Process a single file."""
        try:
            raw_text = load_text_from_file(file_path)
            
            # Skip error documents (API errors, captcha, etc.) - silently
            if is_error_document(raw_text, file_path):
                return None, None  # Skip silently, not an error
            
            text = clean_text(raw_text)
            
            if len(text) < 100:
                return None, "text too short"
            
            if openai_client:
                extraction = parse_with_openai(text, openai_client)
            elif gemini_model:
                time.sleep(12)  # Rate limit for Gemini
                extraction = parse_with_gemini(text, gemini_model)
            else:
                extraction = parse_with_regex(text)
            
            if not extraction:
                return None, "extraction failed"
            
            case = process_result(extraction, file_path.stem, text)
            return case, None
            
        except Exception as e:
            return None, str(e)
    
    # Use parallel processing for OpenAI
    if workers > 1 and openai_client:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_single_file, f): f for f in files}
            
            with tqdm(total=len(files)) as pbar:
                for future in as_completed(futures):
                    file_path = futures[future]
                    try:
                        case, error = future.result()
                        if case:
                            results.append(case)
                        else:
                            errors += 1
                            if fail_fast and error:
                                print(f"\n❌ Error: {file_path.name}: {error}")
                                executor.shutdown(wait=False, cancel_futures=True)
                                break
                    except Exception as e:
                        errors += 1
                        if fail_fast:
                            print(f"\n❌ Error: {file_path.name}: {e}")
                            executor.shutdown(wait=False, cancel_futures=True)
                            break
                    pbar.update(1)
    else:
        # Sequential processing
        for file_path in tqdm(files):
            case, error = process_single_file(file_path)
            if case:
                results.append(case)
            else:
                errors += 1
                if fail_fast and error:
                    print(f"\n❌ Error: {file_path.name}: {error}")
                    break
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Stats
    print(f"\n{'='*50}")
    print(f"PARSING COMPLETE")
    print(f"{'='*50}")
    print(f"Total files: {len(files)}")
    print(f"Successfully parsed: {len(results)}")
    print(f"Errors: {errors}")
    print(f"Output: {output_file}")
    
    if results:
        print(f"\nTotal statute citations: {sum(len(c['statute_ids']) for c in results)}")
        print(f"Unique statutes: {len(set(s for c in results for s in c['statute_ids']))}")
        
        # Outcome distribution
        outcomes = {}
        for c in results:
            o = c.get("outcome") or "UNKNOWN"
            outcomes[o] = outcomes.get(o, 0) + 1
        print("\nOutcome distribution:")
        for o, cnt in sorted(outcomes.items(), key=lambda x: -x[1]):
            print(f"  {o}: {cnt} ({100*cnt/len(results):.1f}%)")



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse legal documents with LLM or regex")
    parser.add_argument("--input", default="data/raw", help="Input directory")
    parser.add_argument("--output", default="data/parsed.json", help="Output file")
    parser.add_argument("--openai", action="store_true", help="Use OpenAI GPT-4o-mini")
    parser.add_argument("--gemini", action="store_true", help="Use Gemini API")
    parser.add_argument("--regex-only", action="store_true", help="Use regex only (no LLM)")
    parser.add_argument("--limit", type=int, help="Limit number of files")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--no-fail-fast", action="store_true", help="Continue on errors")
    
    args = parser.parse_args()
    
    parse_files(
        args.input,
        args.output,
        use_gemini=args.gemini,
        use_openai=args.openai,
        limit=args.limit,
        fail_fast=not args.no_fail_fast,
        workers=args.workers
    )
