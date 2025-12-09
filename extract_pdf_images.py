"""
Extract images from PDF files.

Supports:
- Batch extraction from PDFs folder
- Single PDF file extraction with custom destination
- Output to Extracted_Images folder

Usage:
    python extract_pdf_images.py                           # Extract all PDFs from PDFs/ folder
    python extract_pdf_images.py path/to/file.pdf          # Extract single PDF
    python extract_pdf_images.py path/to/file.pdf path/to/output  # Extract to custom folder
"""

import argparse
import sys
from pathlib import Path
import fitz  # PyMuPDF

# Configuration
DEFAULT_PDF_FOLDER = Path(__file__).parent / "PDFs"
DEFAULT_OUTPUT_FOLDER = Path(__file__).parent / "Extracted_Images"

def extract_images_from_pdf(pdf_path, output_subfolder, pdf_index):
    """
    Extract page images from a single PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        output_subfolder: Subfolder to save images (numbered 1, 2, 3, etc.)
        pdf_index: Sequential index of the PDF (for naming)
        
    Returns:
        Number of images extracted
    """
    try:
        print(f"  📄 Processing: {pdf_path.name}")
        
        # Open PDF document
        pdf_document = fitz.open(str(pdf_path))
        
        if len(pdf_document) == 0:
            print(f"    ⚠️  No pages found in {pdf_path.name}")
            return 0
        
        output_subfolder.mkdir(parents=True, exist_ok=True)
        
        # Process each page
        total_images = 0
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            try:
                # Render page to image (2x zoom for better quality)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                output_path = output_subfolder / f"pdf{pdf_index}_{page_num}.png"
                pix.save(str(output_path))
                total_images += 1
            except Exception as page_error:
                # Try alternative rendering if first attempt fails
                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        output_path = output_subfolder / f"pdf{pdf_index}_{page_num}.png"
                        pix.save(str(output_path))
                    else:  # CMYK or other: convert to RGB
                        pix_rgb = fitz.Pixmap(fitz.csRGB, pix)
                        output_path = output_subfolder / f"pdf{pdf_index}_{page_num}.png"
                        pix_rgb.save(str(output_path))
                    total_images += 1
                except Exception as e:
                    print(f"      ⚠️  Could not render page {page_num}: {str(e)}")
        
        pdf_document.close()
        print(f"    ✅ Extracted {total_images} images from {pdf_path.name}")
        return total_images
        
    except Exception as e:
        print(f"    ❌ Error processing {pdf_path.name}: {str(e)}")
        return 0

def extract_batch(pdf_folder, output_folder):
    """
    Extract images from all PDFs in a folder.
    
    Args:
        pdf_folder: Folder containing PDF files
        output_folder: Base output folder for extracted images
    """
    print(f"\n📂 Scanning folder: {pdf_folder}")
    print(f"📤 Output folder: {output_folder}\n")
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    if not pdf_folder.exists():
        print(f"❌ PDF folder not found: {pdf_folder}")
        return False
    
    # Get all PDF files
    pdf_files = sorted(pdf_folder.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in {pdf_folder}")
        return False
    
    print(f"🔍 Found {len(pdf_files)} PDF files\n")
    
    total_images = 0
    successful = 0
    
    # Process each PDF
    for pdf_file in pdf_files:
        # Create subfolder with PDF name (without extension)
        subfolder_name = pdf_file.stem
        output_subfolder = output_folder / subfolder_name
        
        # Extract images
        images_count = extract_images_from_pdf(pdf_file, output_subfolder, successful + 1)
        
        if images_count > 0:
            total_images += images_count
            successful += 1
        else:
            # Remove empty subfolder
            if output_subfolder.exists() and not list(output_subfolder.iterdir()):
                output_subfolder.rmdir()
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 BATCH EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"  Total PDFs found: {len(pdf_files)}")
    print(f"  Successfully extracted: {successful}")
    print(f"  Total images extracted: {total_images}")
    print(f"  Output folder: {output_folder}")
    print("=" * 70)
    
    return successful > 0


def extract_single(pdf_path, output_folder):
    """
    Extract images from a single PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        output_folder: Output folder for extracted images
    """
    pdf_path = Path(pdf_path)
    output_folder = Path(output_folder)
    
    if not pdf_path.exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return False
    
    if not pdf_path.suffix.lower() == ".pdf":
        print(f"❌ File is not a PDF: {pdf_path}")
        return False
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📄 Single PDF Extraction")
    print("=" * 70)
    print(f"  Input: {pdf_path}")
    print(f"  Output: {output_folder}\n")
    
    # Extract images
    images_count = extract_images_from_pdf(pdf_path, output_folder, 1)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"  PDF: {pdf_path.name}")
    print(f"  Total images extracted: {images_count}")
    print(f"  Output folder: {output_folder}")
    print("=" * 70)
    
    return images_count > 0


def main():
    """Main function with CLI support."""
    parser = argparse.ArgumentParser(
        description="Extract images from PDF files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_pdf_images.py
      Extract all PDFs from 'PDFs' folder to 'Extracted_Images'
  
  python extract_pdf_images.py path/to/file.pdf
      Extract single PDF to 'Extracted_Images'
  
  python extract_pdf_images.py path/to/file.pdf /custom/output
      Extract single PDF to custom output folder
  
  python extract_pdf_images.py --batch /pdf/folder --output /output/folder
      Extract all PDFs from custom folder to custom output
        """
    )
    
    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help="PDF file path (for single extraction) or use --batch flag"
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Output folder for single PDF extraction (default: Extracted_Images)"
    )
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="PDF folder for batch extraction (default: PDFs)"
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Output folder for batch extraction (default: Extracted_Images)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("🖼️  PDF Image Extractor")
    print("=" * 70)
    
    # Batch extraction mode
    if args.batch:
        pdf_folder = Path(args.batch)
        output_folder = Path(args.output_folder) if args.output_folder else DEFAULT_OUTPUT_FOLDER
        success = extract_batch(pdf_folder, output_folder)
        sys.exit(0 if success else 1)
    
    # Single PDF extraction mode
    if args.source:
        pdf_path = Path(args.source)
        output_folder = Path(args.output) if args.output else DEFAULT_OUTPUT_FOLDER
        success = extract_single(pdf_path, output_folder)
        sys.exit(0 if success else 1)
    
    # Default: batch extraction from default PDFs folder
    success = extract_batch(DEFAULT_PDF_FOLDER, DEFAULT_OUTPUT_FOLDER)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
