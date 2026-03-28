# Input Documents Folder

Place all your tender PDFs and scanned documents here.

## Folder Structure

```
input_docs/
├── tender_01.pdf
├── tender_02.pdf
├── sample_tenders/
├── latest_tenders/
└── scanned_documents/
```

## Supported Formats

- **PDF files** (.pdf) - Text and scanned PDFs
- **Images** (.png, .jpg, .jpeg, .tiff) - Scanned documents
- **Multi-page PDFs** - Automatically processed

## How to Use

1. Place your tender documents in this folder (or subfolders)
2. Open the Streamlit app
3. Go to "📤 Upload Tenders" tab
4. Click on "Browse from input_docs/" button
5. Select documents you want to process
6. Click "Process Selected Documents"

## Tips

- Organize documents in subfolders by year or project
- Scan documents at 300 DPI for best OCR results
- Ensure PDFs are not password-protected
- Use descriptive filenames