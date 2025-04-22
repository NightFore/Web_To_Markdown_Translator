# main.py

# Extractor
import os
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from readability import Document

# Translator
import re
import torch
from huggingface_hub import login
from transformers import pipeline, AutoTokenizer

class Extractor:
    def __init__(self, output_folder="output", url=None, html_file="original_page.html", md_file="page_content.md"):
        """
        Initializes the Extractor class
        """
        # Configurations variables
        self.output_folder = output_folder
        self.url = url
        self.html_file = html_file
        self.md_file = md_file

        # Initialize variables
        self.html = ""
        self.soup = None

        # Ensure output folder exists
        os.makedirs(self.output_folder, exist_ok=True)

    def _fetch_page(self):
        """
        Fetch the page from the URL if provided.
        """
        if self.url:
            response = requests.get(self.url)
            response.raise_for_status()
            self.html = response.text
            print(f"Fetched from {self.url} → {self.html_file}")
        else:
            print("No URL provided.")

    def load(self):
        """
        Load the HTML from the local file if it exists. Otherwise, fetch the page from the URL.
        """
        path = os.path.join(self.output_folder, self.html_file)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.html = f.read()
            print(f"Loaded: {self.html_file}")
        elif self.url:
            self._fetch_page()
        else:
            raise FileNotFoundError("No HTML file found and no URL provided.")
        self.soup = BeautifulSoup(self.html, "html.parser")

    def _extract_main_content(self):
        """
        Extracts the main content from the HTML using either a direct selector or readability library.
        """
        main = self.soup.select_one("main, .content, .article, .post-body")
        if main:
            return str(main)
        doc = Document(self.html)
        return doc.summary()

    def _html_to_markdown(self):
        """
        Converts the extracted main HTML content into Markdown format.
        """
        main_html = self._extract_main_content()
        return md(main_html, strip=["script", "style", "noscript"])

    def save_html_file(self):
        """
        Save the raw HTML content to a file.
        """
        path = os.path.join(self.output_folder, self.html_file)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.html)
        print(f"Saved HTML: {self.html_file}")

    def save_markdown_file(self):
        """
        Save the converted Markdown content to a file.
        """
        markdown_content = self._html_to_markdown()
        path = os.path.join(self.output_folder, self.md_file)
        with open(path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        print(f"Saved Markdown: {self.md_file}")

    def save(self):
        """
        Save results for both HTML and Markdown files.
        """
        self.save_html_file()
        self.save_markdown_file()


class Translator:
    def __init__(self, device=None):
        """
        Initialize the Translator class.
        """
        # Default to GPU if available, else CPU
        self.device = device or (0 if torch.cuda.is_available() else -1)

        # Store loaded models
        self.models = {}

        # Automatic Hugging Face login (if not already authenticated)
        self._authenticate()

    @staticmethod
    def _authenticate():
        """
        Authenticate with Hugging Face to access gated models.
        """
        try:
            token = input("Please enter your Hugging Face token: ")
            login(token=token)
        except Exception as e:
            print(f"Error during authentication: {e}")
            print("Please log in manually using huggingface-cli login.")
            print()

    def _load_model(self, model_name):
        """
        Load the model if it hasn't been loaded already.
        """
        if model_name not in self.models:
            # Load model for translation
            print(f"Loading model {model_name}...")
            model = pipeline("translation", model=model_name, device=self.device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            max_length = tokenizer.model_max_length
            self.models[model_name] = {
                "model": model,
                "tokenizer": tokenizer,
                "max_length": max_length
            }

    def _preprocess_text(self, text, model_name):
        """
        Preprocess Markdown text by splitting into paragraphs and lines.
        Ensure no chunk exceeds 0.9 * max_length.
        """
        # Retrieve the model info
        chunks = []
        model_info = self.models[model_name]
        max_length = model_info["max_length"]
        max_chunk_length = int(0.9 * max_length)

        # Split by paragraphs (two or more line breaks)
        paragraphs = re.split(r'\n\s*\n', text.strip())

        # Then split each paragraph by lines (single line breaks)
        for para in paragraphs:
            lines = para.strip().split('\n')
            for line in lines:
                clean_line = line.strip()
                if clean_line:
                    # Split the line into smaller chunks if necessary
                    while len(clean_line) > max_chunk_length:
                        chunk = clean_line[:max_chunk_length]
                        chunks.append(chunk)
                        clean_line = clean_line[max_chunk_length:]
                    # Add remaining part of the line if it fits
                    if clean_line:
                        chunks.append(clean_line)

        return chunks

    def translate(self, text, model_name, max_length=None, temperature=None, top_k=None):
        """
        Translate a given text using the model.
        """
        if not text:
            return text

        # Check if text contains only symbols (non-alphabetical characters)
        if re.match(r'^[^\w\s]+$', text):  # Matches only symbols
            return text

        # Load the model if necessary
        self._load_model(model_name)

        # Retrieve the model info
        model_info = self.models[model_name]
        translator = model_info["model"]
        tokenizer = model_info["tokenizer"]
        max_length = max_length or model_info["max_length"]

        # Prepare the translation parameters
        generation_args = {"max_length": max_length}
        if temperature is not None:
            generation_args["temperature"] = temperature
        if top_k is not None:
            generation_args["do_sample"] = True
            generation_args["top_k"] = top_k

        # Check the length of the input
        tokens = tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) < int(0.9 * max_length):
            # Perform translation
            translation = translator(text, **generation_args)

            return translation[0]['translation_text']
        else:
            # Preprocess and split into chunks
            chunks = self._preprocess_text(text, model_name)
            translated_chunks = []

            # Perform translation
            for chunk in chunks:
                translated = translator(chunk, **generation_args)
                translated_chunks.append(translated[0]["translation_text"])

            return "\n".join(translated_chunks)

class TranslationPipeline:
    def __init__(self, url=None, output_folder="output"):
        """
        Initialize the TranslationPipeline class.
        """
        # Default configurations
        self.model_name = "Helsinki-NLP/opus-mt-ja-en"
        self.html_file = "1_original_page.html"
        self.md_file = "2_page_content.md"
        self.translated_md_file = "3_translated_page_content.md"

        # Initialize components
        self.extractor = Extractor(url=url, output_folder=output_folder, html_file=self.html_file, md_file=self.md_file)
        self.translator = Translator()

    def _translate_md(self, markdown_file, translated_md_file="translated_markdown.md"):
        """
        Translates the Markdown content from the file line by line, skipping code blocks, links,
        and lines starting with symbols (headers, lists, etc.).
        This function will properly handle multi-line code blocks as well, and only translate
        the content of lines that start with symbols (e.g., headers).
        """
        # Load Markdown file
        path = os.path.join(self.extractor.output_folder, markdown_file)
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Initialize an empty list to store translated lines
        translated_lines = []

        # Regex patterns to identify code blocks, links, and symbol-starting lines
        code_block_pattern = r'```'  # Start or end of a code block
        link_pattern = r'!\[.*?\]\(.*?\)|\[[^\]]*\]\([^\)]*\)'  # Match image and regular links
        json_pattern = r'^\s*\{.*\}\s*$'  # Match JSON-like structures
        symbol_starting_pattern = r'^[#\-\*\+\>]+'  # Match lines starting with a symbol like #, -, *, +, >

        in_code_block = False  # Flag to track whether we are inside a code block

        # Translate each line individually
        for line in lines:
            # Check if the line starts or ends a code block
            if re.match(code_block_pattern, line):
                # Toggle code block flag (if we are entering or exiting a code block)
                in_code_block = not in_code_block
                translated_lines.append(line.strip())  # Add the code block start/end as is
            elif in_code_block:
                # Inside a code block, append the line as is without translating
                translated_lines.append(line.strip())
            elif re.match(link_pattern, line) or re.match(json_pattern, line):
                # If it's a link or JSON-like structure, don't translate it
                translated_lines.append(line.strip())
            elif re.match(symbol_starting_pattern, line):
                # If the line starts with a symbol (e.g., header, list), preserve the symbol and translate the content
                # Split the line into the symbol part and the text part
                symbol, content = re.match(r'^[#\-*\+\>]+', line).group(0), line.lstrip(
                    ' #*-+>')  # Remove leading spaces and symbols

                # Translate only the content, and preserve the symbol part
                translated_content = self.translator.translate(content.strip(), model_name=self.model_name)
                translated_line = symbol + ' ' + translated_content
                translated_lines.append(translated_line.strip())
            else:
                # Otherwise, translate the current line
                translated_line = self.translator.translate(line.strip(), model_name=self.model_name)
                translated_lines.append(translated_line)

        # Join all translated lines back together
        translated_content = "\n".join(translated_lines)

        # Save the translated content to a new file
        translated_path = os.path.join(self.extractor.output_folder, translated_md_file)
        with open(translated_path, "w", encoding="utf-8") as f:
            f.write(translated_content)

        print(f"Translated Markdown saved to: {translated_md_file}")

    def run(self):
        """
        Execute the pipeline step-by-step.

        Steps:
        1. Load the HTML content.
        2. Extract and save HTML and Markdown files.
        3. Translate the Markdown content.
        """
        self.extractor.load()
        self.extractor.save()
        self._translate_md(self.extractor.md_file, self.translated_md_file)


if __name__ == "__main__":
    target_url = input("Enter a Japanese article URL to translate: ").strip()
    if target_url:
        translation_pipeline = TranslationPipeline(url=target_url)
        translation_pipeline.run()
    else:
        print("No URL provided.")