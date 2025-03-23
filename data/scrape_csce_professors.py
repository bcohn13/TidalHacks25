from bs4 import BeautifulSoup
import json

with open("csce_faculty.html", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

# Store unique names
professors = []

# Match all <span> tags inside the main profiles section
for span in soup.find_all("span"):
    name = span.get_text(strip=True)
    
    # Very rough filter: skip if it's "Faculty", "Staff", etc.
    if name and len(name.split()) >= 2 and name.lower() not in ["faculty", "leadership", "staff"]:
        if name not in professors:
            professors.append(name)

# Save to JSON
with open("csce_professors.json", "w", encoding="utf-8") as f:
    json.dump(professors, f, indent=2)

print(f"✅ Extracted {len(professors)} professors and saved to csce_professors.json.")
