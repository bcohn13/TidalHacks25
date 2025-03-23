from bs4 import BeautifulSoup
import json

# Load the saved HTML file
with open("csce_courses.html", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

# Extract course titles
courses = []
for block in soup.find_all("div", class_="courseblock"):
    title_tag = block.find("h2", class_="courseblocktitle")
    if title_tag:
        course_title = title_tag.get_text(strip=True)
        if course_title.startswith("CSCE"):
            courses.append(course_title)

# Structure for JSON
course_data = {"CSCE": courses}

# Save to JSON
with open("courses.json", "w", encoding="utf-8") as f:
    json.dump(course_data, f, indent=2)

print(f"✅ Parsed {len(courses)} CSCE courses and saved to courses.json.")
