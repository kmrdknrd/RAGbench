import os
from pathlib import Path
from docling.document_converter import DocumentConverter
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered


class PdfProcessor:
    def __init__(self, converter="marker"):
        """Initialize PDF converter with pre-loaded model"""
        if converter not in ["marker", "docling"]:
            raise ValueError("Invalid converter. Please choose 'marker' or 'docling'.")
        
        self.converter_type = converter
        if converter == "marker":
            self.converter = PdfConverter(artifact_dict=create_model_dict())
        elif converter == "docling":
            self.converter = DocumentConverter()
            self.converter.initialize_pipeline("pdf")
    
    def process_document(self, doc_path, save_path=None):
        """Process document using pre-loaded model"""
        # Check if path is to a single file or a directory
        # Process a single file or all files in a directory
        all_texts = []
        
        # Use the file directly if it's a single file, otherwise get all PDFs in directory
        files = [doc_path] if doc_path.is_file() else list(doc_path.glob("*.pdf"))
        
        if save_path is not None:
            if not os.path.exists(save_path):
                # Create save path if it doesn't exist
                print(f"Creating save directory: {save_path}")
                os.makedirs(save_path)
        
        # Process each file
        for i, file in enumerate(files):
            # Skip if file already processed
            if save_path is not None and os.path.exists(save_path / f"{file.stem}.md"):
                print(f"Skipping {file.stem} because .md already exists")
                # Load .md from disk
                with open(save_path / f"{file.stem}.md", "r") as f:
                    text = f.read()
                all_texts.append(text)
                continue
            
            print(f"Processing {file.stem} ({i+1}/{len(files)})")
            
            if self.converter_type == "marker":
                rendered_doc = self.converter(str(file))
                text, _, images = text_from_rendered(rendered_doc)
            elif self.converter_type == "docling":
                rendered_doc = self.converter.convert(file).document
                text = rendered_doc.export_to_markdown()
            all_texts.append(text)
            
            if save_path is not None:  # save to .md's
                with open(save_path / f"{file.stem}.md", "w") as f:
                    f.write(text)
        
        # For single files, return the text as a list with one item
        return all_texts[0:1] if doc_path.is_file() else all_texts 