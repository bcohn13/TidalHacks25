import os
from google import genai
from google.genai import types


def compare_pdfs():
    client = genai.Client(api_key = os.environ.get('api_key'))

    # Fix file paths with raw strings
    syllabus_pdf_file_path = r"C:\Users\joshe\Downloads\math151Syllabus.pdf"
    test_pdf_file_path = r"C:\Users\joshe\Downloads\151_Exam1A_2022A.pdf"

    # Read both PDF files
    with open(syllabus_pdf_file_path, "rb") as f:
        syllabus_data = f.read()
    with open(test_pdf_file_path, "rb") as f:
        test_data = f.read()

    # Create comparison prompt
    prompt = "Analyze these two documents. The first is a course syllabus and the second is a test. Compare them to determine if the test content aligns with topics covered in the syllabus. Identify which test questions match syllabus topics and which ones (if any) don't appear to be covered in the syllabus."

    # Use both documents in the API call
    doc_data = [
      types.Part.from_bytes(data=syllabus_data, mime_type='application/pdf'),
      types.Part.from_bytes(data=test_data, mime_type='application/pdf')
    ]
    response = client.models.generate_content(
      model="gemini-2.0-flash",
      contents=[
          *doc_data,  # Unpack the list of Parts directly into contents
          prompt      # Add the text prompt
      ])
    return response.text