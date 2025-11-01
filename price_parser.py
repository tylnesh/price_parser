#!/usr/bin/env python3
import base64
import json
import os
import re
import sys
import datetime
from io import BytesIO
from typing import Optional, Tuple, List, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import requests
from pdf2image import convert_from_path
from PIL import Image
import random
import string

# -------------------- Config --------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral-small3.2"  # VLM capable model
REQUEST_TIMEOUT = 550  # seconds
PAGE_DPI = 200         # speed vs readability
CONTEXT_REFRESH_EVERY = 4  # re-prime every N pages to avoid drift

# -------------------- Prompt loaders --------------------
def load_store_prompt(store: str) -> str:
    json_path = os.path.join("models", f"{store}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Nenašiel som konfig: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    schema = cfg.get("schema", {})
    allowed_tags = cfg.get("allowed_tags", [])
    categories = cfg.get("categories", [])
    few_shot = cfg.get("few_shot_examples", [])
    extra = cfg.get("extra_instructions", "")

    parts = []
    parts.append(
        "Si asistent, ktorý extrahuje štruktúrované informácie z obrázkov letákov "
        "slovenských supermarketov."
    )
    parts.append(
        "Pravidlá:\n"
        "1) Výstup musí byť LEN čistý JSON – PLOCHÝ ZOZNAM objektov produktov. ŽIADNY text mimo JSON.\n"
        "2) Rešpektuj schému (názvy kľúčov a typy). Neznáme polia vynechaj.\n"
        "3) Nevymýšľaj položky, varianty ani ceny – spracuj len to, čo je jasne viditeľné na obrázku.\n"
        "4) 'price' a 'original_price' sú DESATINNÉ ČÍSLA (float), bez znaku €.\n"
        "5) 'unit' je napr. 'g','kg','ml','L','kus','balenie'. 'unit_count' je číslo (float).\n"
        "6) Ak vieš prečítať značku, ulož ju do 'brand'.\n"
        "7) Tagy len zo zoznamu povolených; ak sa nehodí nič, vynechaj 'tags'.\n"
        "8) Kategóriu nastav len ak si si istý; inak vynechaj."
    )
    if extra:
        parts.append(f"Ďalšie pokyny:\n{extra}")

    parts.append("Schéma (referencia):")
    parts.append(json.dumps(schema, ensure_ascii=False, indent=2))

    if allowed_tags:
        parts.append("Povolené tagy (používaj iba tieto):")
        parts.append(json.dumps(allowed_tags, ensure_ascii=False))

    if categories:
        parts.append("Povolené kategórie (ak relevantné):")
        parts.append(json.dumps(categories, ensure_ascii=False))

    if few_shot:
        parts.append("Krátke príklady (NEKOPÍRUJ do výsledku, len vzor):")
        for ex in few_shot[:2]:
            parts.append(json.dumps(ex, ensure_ascii=False, indent=2))

    parts.append("Teraz vráť LEN čistý JSON – PLOCHÝ ZOZNAM objektov produktov.")
    return "\n\n".join(parts)

def cover_date_prompt() -> str:
    date = datetime.date.today().isoformat()
    return (
        f"Dnešný dátum je {date}.\n"
        "Z prvej strany slovenského supermarketového letáku vyčítaj rozsah platnosti cien.\n"
        "Vráť LEN čistý JSON:\n"
        "{ \"start_date\": \"YYYY-MM-DD\", \"end_date\": \"YYYY-MM-DD\" }\n"
        "Ak nevieš presne zistiť dátumy, vráť {}."
    )

# -------------------- Image / Ollama helpers --------------------
def image_to_base64(img: Image.Image) -> str:
    buffered = BytesIO()
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def prime_context(store: str) -> List[int]:
    """Prime once with rules/schema; capture KV cache to reuse."""
    priming_prompt = load_store_prompt(store) + "\n\nOdpovedz iba krátkym JSON: {\"ok\":true}"
    payload = {
        "model": MODEL_NAME,
        "prompt": priming_prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_ctx": 8192,
            "repeat_penalty": 1.1,
            "num_predict": 16
        },
        "keep_alive": "45m"
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    base_ctx = data.get("context") or []
    return base_ctx if isinstance(base_ctx, list) else []

def call_ollama_with_image(prompt: str, image_base64: str, base_context: Optional[List[int]] = None) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_ctx": 8192,
            "repeat_penalty": 1.2,
            "num_predict": 768
        },
        "keep_alive": "45m"
    }
    if base_context:
        payload["context"] = list(base_context)  # copy to avoid growth

    resp = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if "response" not in data:
        raise RuntimeError(f"Nečakaná odpoveď z Ollama: {data}")
    return data["response"]

def _extract_json_anywhere(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in [("{", "}"), ("[", "]")]:
        start_idxs = [m.start() for m in re.finditer(re.escape(opener), text)]
        for start in start_idxs:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == opener:
                    depth += 1
                elif text[i] == closer:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            continue
    raise ValueError("Nepodarilo sa nájsť platný JSON vo výstupe modelu.")

# -------------------- Normalization / Flattening --------------------
FLAT_ALLOWED_UNITS = {"g", "kg", "ml", "L", "kus", "balenie", "ks"}

def _parse_price_float(v: object) -> Optional[float]:
    s = str(v).replace("€", "").strip().replace(",", ".")
    m = re.search(r"\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

def _maybe_float(v: object) -> Optional[float]:
    try:
        return float(str(v).replace(",", "."))
    except Exception:
        return None

def _infer_unit_and_count_from_name(name: str) -> Tuple[Optional[str], Optional[float]]:
    low = name.lower()
    # e.g., "500 g", "150 g"
    m = re.search(r"(\d+(?:[\.,]\d+)?)\s*g\b", low)
    if m:
        return "g", float(m.group(1).replace(",", "."))
    m = re.search(r"(\d+(?:[\.,]\d+)?)\s*kg\b", low)
    if m:
        return "kg", float(m.group(1).replace(",", "."))
    # liters
    m = re.search(r"(\d+(?:[\.,]\d+)?)\s*l\b", low)
    if m:
        return "L", float(m.group(1).replace(",", "."))
    # pack sizes like "10 ks" or "10 kusov"
    m = re.search(r"(\d+)\s*(?:ks|kusov|kus)\b", low)
    if m:
        return "kus", float(m.group(1))
    return None, None

def _maybe_package(name: str, unit: Optional[str], unit_count: Optional[float]) -> Optional[str]:
    if not unit or unit_count is None:
        return None
    lbl = "balenie" if "balenie" in name.lower() else "kus"
    # format like "180g/ balenie" or "1L/ kus"
    # trim trailing .0 for ints
    count_str = f"{unit_count:.3f}".rstrip("0").rstrip(".")
    return f"{count_str}{unit}/ {lbl}"

def normalize_and_flatten(parsed: object) -> List[dict]:
    """
    Flatten any 'products' nesting and normalize fields to NEW SCHEMA:
      - price -> float
      - original_price -> float
      - unit/unit_count inferred if missing
      - package auto-constructed if missing and unit+count present
      - tags -> list[str]
      - drop items without name or price<=0
    """
    flat: List = []

    def walk(x):
        if isinstance(x, dict):
            if "products" in x and isinstance(x["products"], list):
                for it in x["products"]:
                    walk(it)
            else:
                flat.append(x)
        elif isinstance(x, list):
            for it in x:
                walk(it)

    walk(parsed)

    cleaned: List[dict] = []
    for p in flat:
        if not isinstance(p, dict):
            continue

        name = str(p.get("name") or "").strip()
        if not name:
            continue

        # price as float
        price = _parse_price_float(p.get("price"))
        if price is None or price <= 0:
            continue
        p["price"] = round(price, 2)

        # original_price as float (optional)
        if "original_price" in p:
            op = _parse_price_float(p.get("original_price"))
            if op is None:
                p.pop("original_price", None)
            else:
                p["original_price"] = round(op, 2)

        # unit / unit_count
        unit = str(p.get("unit")).strip() if p.get("unit") else None
        unit_count = _maybe_float(p.get("unit_count")) if p.get("unit_count") is not None else None
        if unit and unit == "ks":
            unit = "kus"
        if unit and unit not in FLAT_ALLOWED_UNITS:
            unit = unit  # keep as-is but we’ll try to infer better
        if not unit or unit_count is None:
            inf_u, inf_c = _infer_unit_and_count_from_name(name)
            unit = unit or inf_u
            unit_count = unit_count if unit_count is not None else inf_c
        if unit:
            p["unit"] = "kus" if unit == "ks" else unit
        if unit_count is not None:
            p["unit_count"] = float(unit_count)

        # package: auto if missing and we know unit + count
        if not p.get("package"):
            pkg = _maybe_package(name, p.get("unit"), p.get("unit_count"))
            if pkg:
                p["package"] = pkg

        # tags
        if "tags" in p and isinstance(p["tags"], list):
            p["tags"] = [t.strip() for t in p["tags"] if isinstance(t, str) and t.strip()]
        elif "tags" in p and isinstance(p["tags"], str):
            p["tags"] = [t.strip() for t in p["tags"].split(",") if t.strip()]

        # brand stays as provided if present (string)

        cleaned.append(p)

    return cleaned

# -------------------- Price audit + filtering (float-based) --------------------
def audit_prices_on_page(img: Image.Image, base_context: Optional[List[int]]) -> Set[float]:
    """
    Ask model to list visible numeric prices on the page; returns set of floats rounded to 2 decimals.
    """
    img_b64 = image_to_base64(img)
    prompt = (
        "Na obrázku vyhľadaj VŠETKY JASNE VIDITEĽNÉ CENY ako čísla (napr. '16.49', '1.29', '0.59'). "
        "Ignoruj slogany bez ceny. Vráť LEN JSON objekt:\n"
        "{ \"prices\": [\"<cena1>\", \"<cena2>\"] }\n"
        "Ceny normalizuj s bodkou, bez symbolu €."
    )
    resp = call_ollama_with_image(prompt, img_b64, base_context=base_context)
    try:
        data = _extract_json_anywhere(resp)
        prices = data.get("prices", []) if isinstance(data, dict) else []
    except Exception:
        prices = []

    out: Set[float] = set()
    for p in prices:
        val = _parse_price_float(p)
        if val is not None and val > 0:
            out.add(round(val, 2))
    return out

def filter_products_by_prices(products: List[dict], allowed_prices: Set[float]) -> List[dict]:
    if not allowed_prices:
        return []  # no visible price -> don't fabricate anything
    kept = []
    for p in products:
        price_val = _parse_price_float(p.get("price"))
        if price_val is None:
            continue
        price_val = round(price_val, 2)
        if price_val in allowed_prices:
            p["price"] = price_val
            kept.append(p)
    return kept

# -------------------- Extraction helpers --------------------
def extract_date_range_from_cover(cover_img: Image.Image, base_context: Optional[List[int]]) -> Tuple[Optional[str], Optional[str]]:
    print("🗓️  Zisťujem dátumový rozsah z 1. strany…")
    img_b64 = image_to_base64(cover_img)
    try:
        raw = call_ollama_with_image(cover_date_prompt(), img_b64, base_context=base_context)
        data = _extract_json_anywhere(raw)
        if isinstance(data, dict):
            sd = data.get("start_date") or None
            ed = data.get("end_date") or None
            iso_re = r"^\d{4}-\d{2}-\d{2}$"
            if sd and not re.match(iso_re, sd):
                sd = None
            if ed and not re.match(iso_re, ed):
                ed = None
            if sd or ed:
                print(f"   ➜ rozsah: {sd or '?'} → {ed or '?'}")
                return sd, ed
    except Exception as e:
        print(f"   ⚠️ Nepodarilo sa získať dátumy: {e}")
    print("   ⚠️ Dátumy sa nepodarilo spoľahlivo určiť.")
    return None, None

def extract_products_from_image(img: Image.Image, page_prompt: str, base_context: Optional[List[int]]) -> List[dict]:
    img_b64 = image_to_base64(img)
    response_text = call_ollama_with_image(page_prompt, img_b64, base_context=base_context)
    parsed = _extract_json_anywhere(response_text)
    items = normalize_and_flatten(parsed)

    # If shape is still wrong, retry once with a harder directive
    bad_shape = not isinstance(parsed, list) or any(isinstance(x, dict) and "products" in x for x in (parsed if isinstance(parsed, list) else []))
    if bad_shape:
        hard_page_prompt = (
            "Vráť výhradne JSON – PLOCHÝ ZOZNAM objektov produktov (array). "
            "Bez obalu, bez 'products', bez textu. Len [ {...}, {...} ]. "
            "Typy: price/original_price sú čísla (float)."
        )
        response_text = call_ollama_with_image(hard_page_prompt, img_b64, base_context=base_context)
        parsed2 = _extract_json_anywhere(response_text)
        items2 = normalize_and_flatten(parsed2)
        if len(items2) >= len(items):
            items = items2

    return items

def apply_date_range_to_products(products: list, start_date: Optional[str], end_date: Optional[str]) -> list:
    for p in products:
        if isinstance(p, dict):
            if start_date and not p.get("start_date"):
                p["start_date"] = start_date
            if end_date and not p.get("end_date"):
                p["end_date"] = end_date
    return products

def sanitize_filename_part(s: str) -> str:
    return re.sub(r"[^a-z0-9_\-]", "", s.strip().lower().replace(" ", "_"))

def build_default_output_filename(store: str, start_date: Optional[str], end_date: Optional[str]) -> str:
    store_part = sanitize_filename_part(store)
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    if start_date and end_date:
        return f"output/{store_part}_{start_date}_to_{end_date}_{suffix}.json"
    return f"output/{store_part}_{suffix}_products.json"

# -------------------- Parallel extraction with context refresh --------------------
def _process_single_page(page_index: int,
                         img: Image.Image,
                         page_prompt: str,
                         start_date: Optional[str],
                         end_date: Optional[str],
                         base_context: Optional[List[int]]) -> List[dict]:
    try:
        # 0) Find which prices are actually visible on the page
        visible_prices = audit_prices_on_page(img, base_context)

        # 1) Extract candidate products
        items = extract_products_from_image(img, page_prompt, base_context=base_context)

        # 2) Keep only products whose price matches a visible price
        items = filter_products_by_prices(items, visible_prices)

        # 3) Apply dates
        items = apply_date_range_to_products(items, start_date, end_date)
        return items
    except Exception as e:
        print(f"   ❌ Chyba na strane {page_index}: {e}")
        return []

def extract_from_pdf(pdf_path: str, store: str, out_path: str, start_date: str, end_date: str, poppler_path: str | None = None) -> list:
    print(f"📄 Spracovávam PDF: {pdf_path}")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Súbor neexistuje: {pdf_path}")

    pages = convert_from_path(pdf_path, dpi=PAGE_DPI, poppler_path=poppler_path)
    if not pages:
        return []

    print("🧠 Priming modelu (jednorazové)…")
    base_ctx = prime_context(store)

    # Ultra-short page prompt (rules live in KV cache)
    page_prompt = (
        "Vráť LEN JSON – PLOCHÝ ZOZNAM objektov produktov, KTORÉ MAJÚ NA OBRÁZKU VIDITEĽNÚ ČÍSELNÚ CENU. "
        "Nevkladaj 'products' obal, len [ {...}, {...} ]. "
        "Nevymýšľaj varianty; extrahuj len položky s jasnou cenou. "
        "Typy: price/original_price sú čísla (float); unit_count je číslo (float)."
    )

    all_items: list = []
    tmp_path = out_path + ".tmp"
    lock = threading.Lock()
    max_workers = min(4, (os.cpu_count() or 4))
    print(f"🚀 Paralelné spracovanie strán (workers={max_workers})…")

    # Process in chunks to refresh context regularly
    for chunk_start in range(0, len(pages), CONTEXT_REFRESH_EVERY):
        chunk = pages[chunk_start : chunk_start + CONTEXT_REFRESH_EVERY]
        if chunk_start > 0:
            print(f"🔄 Refreshujem kontext po {chunk_start} stranách…")
            base_ctx = prime_context(store)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {}
            for offset, img in enumerate(chunk, start=chunk_start + 1):
                print(f"  • frontujem stranu {offset}/{len(pages)}")
                fut = ex.submit(_process_single_page, offset, img, page_prompt, start_date, end_date, base_ctx)
                futures[fut] = offset

            for fut in as_completed(futures):
                page_no = futures[fut]
                items = fut.result() or []
                with lock:
                    all_items.extend(items)
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        json.dump(all_items, f, indent=2, ensure_ascii=False)
                print(f"   ✅ dokončená strana {page_no}; priebežný počet položiek: {len(all_items)}")

    return all_items

# -------------------- CLI --------------------
def main(pdf_path: str, store: str, out_path: Optional[str] = None, poppler_path: str | None = None):
    # Build filename using quick cover pass + priming to capture dates
    tmp_pages = convert_from_path(pdf_path, dpi=PAGE_DPI, first_page=1, last_page=1)
    base_ctx = prime_context(store)
    sd, ed = extract_date_range_from_cover(tmp_pages[0], base_context=base_ctx) if tmp_pages else (None, None)

    if not out_path:
        out_path = build_default_output_filename(store, sd, ed)

    extracted = extract_from_pdf(pdf_path, store, out_path, sd, ed, poppler_path=poppler_path)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(extracted, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Hotovo! Extrahovaných položiek: {len(extracted)} -> {out_path}")

    tmp_path = out_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Použitie: python price_parser.py letak.pdf obchod [vystup.json]")
        sys.exit(1)
    pdf = sys.argv[1]
    store = sys.argv[2]
    out = sys.argv[3] if len(sys.argv) >= 4 else None
    main(pdf, store, out_path=out, poppler_path=None)
