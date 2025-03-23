from bs4 import BeautifulSoup
import json

with open("tamu_catalog.html", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

subjects = []

# Find all links that go to /undergraduate/course-descriptions/...
for link in soup.find_all("a", href=True):
    href = link["href"]
    if "/undergraduate/course-descriptions/" in href:
        parts = href.strip("/").split("/")
        if len(parts) >= 3:
            code = parts[-1].upper()
            name = link.get_text(strip=True)
            subjects.append({
                "code": code,
                "name": name
            })

# Save to JSON
with open("subjects.json", "w", encoding="utf-8") as f:
    json.dump(subjects, f, indent=2)

print(f"✅ Extracted {len(subjects)} subjects and saved to subjects.json.")
