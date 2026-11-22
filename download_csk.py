
from pathlib import Path
import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent

OUT = BASE_DIR / "data" / "rag_documents" / "cyber_swachhta_kendra"
OUT.mkdir(parents=True, exist_ok=True)

pages = {
    "mirai_botnet": "https://www.csk.gov.in/alerts/mirai.html",

    "iot_ip_camera_botnets": "https://www.csk.gov.in/alerts/IOT_Botnets_Targeting_Vulnerable_IP_cameras.html",

    "botenago_malware": "https://www.csk.gov.in/alerts/BotenaGo.html",
}

for name, url in pages.items():
    try:
        print(f"\nDownloading: {name}")

        response = requests.get(
            url,
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        main = soup.find("main") or soup.body or soup

        for tag in main.find_all(
            ["script", "style", "nav", "footer"]
        ):
            tag.decompose()

        title = soup.title.get_text(strip=True) if soup.title else name
        content = main.get_text(separator="\n", strip=True)

        if not content:
            raise ValueError("No webpage text extracted")

        file_path = OUT / f"{name}.txt"

        file_path.write_text(
            f"Title: {title}\n"
            f"Source: {url}\n\n"
            f"{content}",
            encoding="utf-8"
        )

        print(f"Saved: {file_path.name}")

    except Exception as e:
        print(f"ERROR: {name}: {e}")

print("\nDownload process completed.")
print(f"Files saved in: {OUT}")
