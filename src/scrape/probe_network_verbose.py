import json
from datetime import datetime
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path("data/raw/_probe")
OUT.mkdir(parents=True, exist_ok=True)

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT / f"network_verbose_{ts}.jsonl"

    print("Log dosyası:", out_path)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        def log(obj):
            # hem dosyaya hem ekrana kısa özet
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        def on_request(req):
            obj = {
                "type": "request",
                "method": req.method,
                "url": req.url,
                "resource_type": req.resource_type,
                "post_data": req.post_data,
            }
            log(obj)
            # Terminalde çok akmasın diye sadece xhr/fetch/other yazdır
            if req.resource_type in ("xhr", "fetch", "other"):
                print(f"REQ [{req.resource_type}] {req.method} {req.url}")

        def on_response(res):
            req = res.request
            obj = {
                "type": "response",
                "status": res.status,
                "url": res.url,
                "resource_type": req.resource_type,
                "method": req.method,
            }
            log(obj)
            if req.resource_type in ("xhr", "fetch", "other"):
                print(f"RES [{req.resource_type}] {res.status} {res.url}")

        page.on("request", on_request)
        page.on("response", on_response)

        page.goto("https://karararama.yargitay.gov.tr/", wait_until="domcontentloaded")

        print("\n--- MANUEL ---")
        print("1) Sitede 'iş hukuk' ara")
        print("2) Sonuçlar gelince 2. sayfaya geç (spinner dönsün)")
        print("3) Terminalde REQ/RES satırları akıyor mu bak")
        input("\nBittiğinde ENTER...\n")

        browser.close()

    print("✅ Bitti. Log:", out_path)

if __name__ == "__main__":
    main()
