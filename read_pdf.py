from pypdf import PdfReader

try:
    reader = PdfReader("ICEIC2026_final.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    with open("paper_content.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("PDF text extracted to paper_content.txt")
except Exception as e:
    print(f"Error reading PDF: {e}")
