"""
Çok Metrikli Bağımsız JSON Ayrıştırıcı.

Tasarım Prensibi:
Metrikleri birbirinden tamamen BAĞIMSIZ ölçer:
1. is_pure_json: Model prompt kuralına uyup el değmeden saf { ... } JSON üretti mi?
2. is_valid_syntax: Markdown bloğu soyulduğunda içerideki JSON sözdizimsel olarak geçerli mi?
3. parsed_data: Çıkarılan JSON sözlüğü (kategori, sentiment ve entity değerlendirmesi için)
4. format_type: 'pure_json', 'markdown_fence', 'invalid_syntax', 'empty'
"""

import json
import re
from typing import Dict, Any, Optional, NamedTuple


class ParseResult(NamedTuple):
    is_pure_json: bool       # Hocanın 1. Metriği: Ham çıktıya doğrudan json.loads
    is_valid_syntax: bool    # Hocanın 2. Metriği: Temizleme sonrasında geçerli sözdizimi
    is_schema_valid: bool    # Hocanın 3. Metriği: Zorunlu alanlar ve tiplerin doğruluğu
    format_type: str         # 'pure_json', 'markdown_fence', 'text_with_json', 'invalid_syntax', 'empty'
    data: Optional[Dict[str, Any]]
    error: Optional[str]


def validate_schema(data: Any) -> bool:
    """
    Şema geçerlilik kontrolü (Hocanın 3. zorunlu değerlendirme metriği):
    1. data bir sözlük (dict) olmalı.
    2. 'category', 'sentiment' ve 'entities' alanları mevcut olmalı.
    3. 'category' ve 'sentiment' string tipinde olmalı.
    4. 'entities' bir liste olmalı ve tüm elemanları string olmalı.
    """
    if not isinstance(data, dict):
        return False
    required = {"category", "sentiment", "entities"}
    if not required.issubset(data.keys()):
        return False
    if not isinstance(data.get("category"), str):
        return False
    if not isinstance(data.get("sentiment"), str):
        return False
    entities = data.get("entities")
    if not isinstance(entities, list) or not all(isinstance(e, str) for e in entities):
        return False
    return True


def parse_llm_json(raw_output: str) -> ParseResult:
    """
    Ham LLM çıktısını analiz eder ve metrikleri birbirinden bağımsız ayrıştırır.
    """
    if not raw_output or not raw_output.strip():
        return ParseResult(
            is_pure_json=False,
            is_valid_syntax=False,
            is_schema_valid=False,
            format_type="empty",
            data=None,
            error="Boş yanıt"
        )

    text = raw_output.strip()

    # 1. Aşama: El değmemiş Saf JSON testi (Raw JSON Parsing Success)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return ParseResult(
                is_pure_json=True,
                is_valid_syntax=True,
                is_schema_valid=validate_schema(data),
                format_type="pure_json",
                data=data,
                error=None
            )
    except Exception:
        pass

    # 2. Aşama: Markdown fence soyma testi (Model ```json kullandıysa)
    fence_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(fence_pattern, text)
    if match:
        extracted = match.group(1).strip()
        try:
            data = json.loads(extracted)
            if isinstance(data, dict):
                return ParseResult(
                    is_pure_json=False,
                    is_valid_syntax=True,
                    is_schema_valid=validate_schema(data),
                    format_type="markdown_fence",
                    data=data,
                    error=None
                )
        except json.JSONDecodeError as e:
            return ParseResult(
                is_pure_json=False,
                is_valid_syntax=False,
                is_schema_valid=False,
                format_type="invalid_syntax",
                data=None,
                error=f"Markdown içi JSONDecodeError: {e}"
            )

    # 3. Aşama: Herhangi bir { ... } bloğu var mı? (Regex ile kurtarma)
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        try:
            data = json.loads(brace_match.group(0).strip())
            if isinstance(data, dict):
                return ParseResult(
                    is_pure_json=False,
                    is_valid_syntax=True,
                    is_schema_valid=validate_schema(data),
                    format_type="text_with_json",
                    data=data,
                    error=None
                )
        except json.JSONDecodeError:
            pass

    return ParseResult(
        is_pure_json=False,
        is_valid_syntax=False,
        is_schema_valid=False,
        format_type="invalid_syntax",
        data=None,
        error="Geçersiz JSON sözdizimi"
    )

