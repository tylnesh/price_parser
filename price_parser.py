import base64
import json
import requests
from io import BytesIO
from pdf2image import convert_from_path
from PIL import Image

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral-small3.2"
OUTPUT_JSON = "produkty_z_letaku.json"

PROMPT = """
Si asistent, ktorý extrahuje štruktúrované informácie z obrázkov letákov slovenských supermarketov.
Z obrázka vyextrahuj JSON zoznam produktov. Každý produkt by mal obsahovať:

- name (názov, string)
- price (cena po zľave, float alebo string, napr. "2 za 3.99")
- original_price (nepovinné, pôvodná cena pred zľavou, float)
- unit (nepovinné, napr. "kg", "L", "balenie", atď.)
- discount_description (nepovinné, stručný opis zľavy, napr. "-33%", "akcia", "1+1 zadarmo")
- lidl_plus (boolean, default false)

Výstup musí byť **len čistý JSON**. Bez vysvetlenia, popisu, ani formátovania navyše.
"""

def image_to_base64(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def call_ollama_with_image(prompt: str, image_base64: str) -> str:
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False,
    })
    response.raise_for_status()
    return response.json()["response"]

def extract_from_pdf(pdf_path: str) -> list:
    print(f"📄 Spracovávam PDF: {pdf_path}")
    pages = convert_from_path(pdf_path, dpi=200)
    all_items = []

    for i, img in enumerate(pages):
        print(f"🔍 Strana {i+1}/{len(pages)}")
        img_b64 = image_to_base64(img)
        response_text = call_ollama_with_image(PROMPT, img_b64)

        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, list):
                all_items.extend(parsed)
            else:
                print(f"⚠️ Strana {i+1}: Výstup nie je zoznam")
        except json.JSONDecodeError:
            print(f"❌ Nepodarilo sa dekódovať JSON zo strany {i+1}")
            print(response_text)

    return all_items

def main(pdf_path: str):
    extracted = extract_from_pdf(pdf_path)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(extracted, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Hotovo! Extrahovaných položiek: {len(extracted)} -> {OUTPUT_JSON}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Použitie: python price_parser.py letak.pdf")
    else:
        main(sys.argv[1])
