from pdf2image import convert_from_path
import os

# Paths
BASE_DIR = "exam_ocr_project"
PDF_DIR = os.path.join(BASE_DIR, "data_pdfs/Student_Pdf")
IMG_DIR = os.path.join(BASE_DIR, "dataset", "page_images")

os.makedirs(IMG_DIR, exist_ok=True)

def convert_all_pdfs():
    # Loop through all PDFs in data_pdfs
    for pdf_file in sorted(os.listdir(PDF_DIR)):
        if not pdf_file.lower().endswith(".pdf"):
            continue

        student_name = pdf_file.replace(".pdf", "")
        student_dir = os.path.join(IMG_DIR, student_name)
        os.makedirs(student_dir, exist_ok=True)

        pdf_path = os.path.join(PDF_DIR, pdf_file)
        print(f" Converting {pdf_file}...")

        # Convert PDF → PNGs
        pages = convert_from_path(pdf_path, dpi=200)
        for i, page in enumerate(pages):
            page_path = os.path.join(student_dir, f"page_{i}.png")
            page.save(page_path, "PNG")

        print(f" Saved {len(pages)} pages to {student_dir}")

if __name__ == "__main__":
    convert_all_pdfs()
    print(" All PDFs converted successfully!")
