import os
import chardet
from fpdf import FPDF

def txt_to_pdf_folder():
    # Get a list of all TXT files in the "textDocs" folder
    txt_files = [f for f in os.listdir("textDocs") if f.endswith(".txt")]

    # Loop through the TXT files and convert each one to a PDF
    for txt_file in txt_files:
        # Get the input and output file paths
        input_file = os.path.join("textDocs", txt_file)
        output_file = os.path.join("textDocs", os.path.splitext(txt_file)[0] + ".pdf")

        # Open the input file and read its contents, detecting the encoding automatically
        with open(input_file, "rb") as f:
            encoding = chardet.detect(f.read())["encoding"]
        with open(input_file, "r", encoding=encoding) as f:
            txt = f.read()

        # Create a new PDF document and add a page
        pdf = FPDF()
        pdf.add_page()

        # Set the font and size for the text
        pdf.set_font("Arial", size=12)

        # Write the text to the PDF document
        pdf.write(5, txt)

        # Save the PDF document to a file using UTF-8 encoding
        pdf.output(output_file, "F")


txt_to_pdf_folder()