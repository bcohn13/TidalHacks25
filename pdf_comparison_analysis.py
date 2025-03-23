import os
from google import genai
from google.genai import types

def analyze_user_pdf(user_pdf_path=None, comparison_pdf_path=None, custom_prompt=None):
  """
  Analyze a user-provided PDF, optionally comparing it with another PDF.
  
  Args:
    user_pdf_path (str): Path to the user's PDF file
    comparison_pdf_path (str, optional): Path to a PDF to compare with
    custom_prompt (str, optional): Custom prompt for the analysis
  
  Returns:
    str: Analysis result text
  """
  try:
    # Check if user_pdf_path is provided
    if user_pdf_path is None:
      return "No pdf file provided"
      
    client = genai.Client(api_key=os.environ.get('api_key'))
    
    # Check if file exists
    if not os.path.exists(user_pdf_path):
      return f"Error: File not found at {user_pdf_path}"
      
    # Read user PDF file
    with open(user_pdf_path, "rb") as f:
      user_data = f.read()
    
    # Create document parts list
    doc_data = [types.Part.from_bytes(data=user_data, mime_type='application/pdf')]
    
    if comparison_pdf_path:
      if not os.path.exists(comparison_pdf_path):
        return f"Error: Comparison file not found at {comparison_pdf_path}"
        
      # Read comparison PDF if provided
      with open(comparison_pdf_path, "rb") as f:
        comparison_data = f.read()
      doc_data.append(types.Part.from_bytes(data=comparison_data, mime_type='application/pdf'))
      prompt = custom_prompt or "Compare these two documents and identify key similarities and differences."
    else:
      prompt = custom_prompt or "Analyze this document and provide a detailed summary."
    
    # Generate content
    response = client.models.generate_content(
      model="gemini-2.0-flash",
      contents=[*doc_data, prompt]
    )
    
    return response.text
  except Exception as e:
    return f"Error analyzing PDF: {str(e)}"
