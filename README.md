# Web to Markdown Translator

A tool that extracts the main article content from a Japanese website, converts it to Markdown, and translates it into English using a simple Hugging Face's `Helsinki-NLP/opus-mt-ja-en` model. Please note that this tool isn't perfect and may have some issues with handling specific formatting or symbols.

## Table of Contents
1. [Setup Instructions](#setup-instructions)  
2. [Project Structure](#project-structure)  
3. [Key Features](#key-features)  
4. [License](#license)  

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/NightFore/Web_To_Markdown_Translator.git
cd Web_To_Markdown_Translator
```

### 2. Install Dependencies

Install all required Python packages:

```bash
# 1. HTML parsing and markdown conversion libraries
pip install markdownify readability-lxml lxml_html_clean

# 2. Translation-related utilities
pip install transformers sentencepiece sacremoses accelerate

# 3. PyTorch and required CUDA packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Note**: The `torch` and related packages are installed for CUDA 12.4. If you're using a different version of CUDA, you may need to modify the installation command.

### 3. Run the Script

The script fetches a webpage, extracts its main content as Markdown, and translates it to English.

```bash
python main.py
```

## Project Structure

```
Web_To_Markdown_Translator/
│
├── main.py                         # Main script to run
├── readme.md                       # This README file
├── requirements.txt                # Project dependencies (optional)
├── output/                         # Folder containing HTML, Markdown, and translated files
│   ├── 1_original_page.html
│   ├── 2_page_content.md
│   └── 3_translated_page_content.md
```

## Key Features

- 🌐 **Fetch & Parse HTML**: Load a webpage via URL or from a local HTML file.
- 📝 **Extract Main Content**: Strips out unnecessary scripts, ads, and UI elements using Readability and custom selectors.
- 🔁 **Convert to Markdown**: Uses `markdownify` to create readable Markdown files from HTML.
- 🌍 **Translate with Transformers**: Translate extracted content using Hugging Face's `Helsinki-NLP/opus-mt-ja-en` or other supported models.
- 💾 **Save Outputs**: HTML, Markdown, and translated Markdown are saved for later use.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
