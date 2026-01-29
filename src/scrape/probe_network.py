import json
from datetime import datetime
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path("data/raw/_probe")
OUT.mkdir(parents=True, exist_ok=True)

TARGET_HOST = "karararama.yargitay.gov.tr"

def main():
    logs = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT / f"network_{ts}.jsonl"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # ilk seferde False
        page = browser.new_page()

        def on_request(req):
            if TARGET_HOST not in req.url:
                return
            if req.resource_type not in ("xhr", "fetch"):
                return
            logs.append({
                "type": "request",
                "method": req.method,
                "url": req.url,
                "resource_type": req.resource_type,
                "headers": {k: v for k, v in req.headers.items() if k.lower() in ("content-type", "accept", "referer")},
                "post_data": req.post_data,
            })

        def on_response(res):
            if TARGET_HOST not in res.url:
                return
            req = res.request
            if req.resource_type not in ("xhr", "fetch"):
                return
            item = {
                "type": "response",
                "status": res.status,
                "url": res.url,
                "method": req.method,
                "resource_type": req.resource_type,
            }
            logs.append(item)

        page.on("request", on_request)
        page.on("response", on_response)

        page.goto(f"https://{TARGET_HOST}/", wait_until="networkidle")

        print("\n--- MANUEL ADIM ---")
        print("1) Açılan sayfada arama kutusuna: iş hukuk")
        print("2) Ara'ya bas")
        print("3) Sonuçlar gelince 1-2 sayfa ileri/geri yap")
        print("4) Sonra buraya dön ve ENTER'a bas\n")

        input("Devam etmek için ENTER...")

        # logları kaydet
        with out_path.open("w", encoding="utf-8") as f:
            for x in logs:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

        print(f"\n✅ Network log kaydedildi: {out_path}")
        browser.close()

if __name__ == "__main__":
    main()
