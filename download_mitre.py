
from pathlib import Path
import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "data" / "rag_documents" / "mitre_attack"
OUT.mkdir(parents=True, exist_ok=True)

pages = {
    "T1498_network_denial_of_service": "https://attack.mitre.org/techniques/T1498/",
    "T1498_001_direct_network_flood": "https://attack.mitre.org/techniques/T1498/001/",
    "T1498_002_reflection_amplification": "https://attack.mitre.org/techniques/T1498/002/",
    "T1499_endpoint_denial_of_service": "https://attack.mitre.org/techniques/T1499/",
    "T1583_005_acquire_botnet": "https://attack.mitre.org/techniques/T1583/005/",
    "T1584_005_compromise_botnet": "https://attack.mitre.org/techniques/T1584/005/",
}

for name, url in pages.items():
    try:
        print(f"Downloading: {name}")

        response = requests.get(
            url,
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        main = soup.find("main") or soup.body

        if main is None:
            raise ValueError("Could not find webpage content")

        for tag in main.find_all(["script", "style", "nav", "footer"]):
            tag.decompose()

        content = main.get_text(separator="\n", strip=True)

        if not content:
            raise ValueError("No text content extracted")

        file_path = OUT / f"{name}.txt"
        file_path.write_text(
            f"Source: {url}\n\n{content}",
            encoding="utf-8"
        )

        print(f"Saved: {file_path.name}")

    except Exception as e:
        print(f"ERROR downloading {name}: {e}")

print("\nDownload process completed.")
print(f"Output folder: {OUT}")
