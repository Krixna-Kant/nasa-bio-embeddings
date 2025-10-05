import os
import json
import re
import pdfplumber
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI
from pinecone import Pinecone
import tiktoken
from pinecone import ServerlessSpec

# --- Load keys ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# --- Initialize clients ---
genai.configure(api_key=GOOGLE_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "nasa-bio-embeddings-new"
if index_name not in [i["name"] for i in pc.list_indexes()]:
    # pc.create_index(name=index_name, dimension=1536, metric="cosine")
    pc.create_index(
    name=index_name,
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
else:
    print(f"Index '{index_name}' already exists.")
    
index = pc.Index(index_name)

encoding = tiktoken.get_encoding("cl100k_base")

# --- Load CSV metadata ---
csv_path = os.path.join(os.path.dirname(__file__), "../data/csv/pinecone_documents.csv")
doc_metadata = pd.read_csv(csv_path)

# Helper function to find metadata by filename
def get_doc_metadata(filename):
    row = doc_metadata[doc_metadata["filename"] == filename]
    if row.empty:
        return None
    return {
        "title": row.iloc[0]["title"],
        "url": row.iloc[0]["URL"]
    }

# --- Helpers ---
def chunk_text(text, max_tokens=750, overlap=150):
    tokens = encoding.encode(text)
    chunks, start = [], 0
    while start < len(tokens):
        end = start + max_tokens
        chunks.append(encoding.decode(tokens[start:end]))
        start += (max_tokens - overlap)
    return chunks

def extract_pdf_text(pdf_path):
    """Extract text from a PDF file."""
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)

# --- Regex-based metadata extraction fallback ---
def extract_title(text):
    match = re.search(r"(?:(?:Article|Paper|Research)\s*)?([\w\s,:;'\-\(\)\[\]/&]+)\n", text)
    if match:
        title = match.group(1).strip()
        if len(title.split()) > 3 and not title.lower().startswith("received"):
            return title
    return ""

def extract_authors(text):
    author_block = re.search(r"(?:[A-Z][\w\.\-\s]+(?:,|and|\d))+[A-Z][\w\.\-\s]+", text)
    if author_block:
        raw = author_block.group(0)
        authors = re.split(r",|and", raw)
        authors = [a.strip() for a in authors if len(a.strip()) > 2 and "@" not in a]
        return authors
    return []

def extract_year(text):
    match = re.search(r"(?:19|20)\d{2}", text)
    if match:
        return match.group(0)
    return ""

# --- Gemini section & metadata extraction ---
def gemini_extract_sections(text):
    prompt = f"""
    You are an expert in scientific literature analysis.
    Segment the following paper into JSON with:
    {{
      "paper_id": "",
      "title": "",
      "authors": [],
      "year": "",
      "doi": "",
      "sections": {{
         "Abstract": "",
         "Introduction": "",
         "Methods": "",
         "Results": "",
         "Discussion": "",
         "Conclusion": "",
         "References": ""
      }}
    }}
    Fill missing fields as "" or [].
    Return only valid JSON.
    Text: {text[:20000]}
    """
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    try:
        return json.loads(response.text)
    except:
        return {"sections": {"Body": text}}

# --- GPT-4o-mini summarization ---
def summarize_section(text, section_name):
    prompt = f"Summarize this {section_name} section concisely:\n\n{text[:4000]}"
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content.strip()

# --- Upload to Pinecone ---
def upload_to_pinecone(filename, metadata, title, url):
    sections = metadata.get("sections", {})
    paper_meta = {
        "paper_id": filename,
        "title": title,
        "url": url,
        "authors": metadata.get("authors", []),
        "year": metadata.get("year", ""),
        "doi": metadata.get("doi", "")
    }

    for sname, stext in sections.items():
        if not stext or not stext.strip():
            continue
        summary = summarize_section(stext, sname)
        chunks = chunk_text(summary)
        for i, chunk in enumerate(chunks):
            emb = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk
            ).data[0].embedding

            meta = {
                "paper_id": filename,
                "type": "section_summary",
                "section": sname,
                "text": chunk,
                "title": title,
                "url": url,
                "authors": paper_meta["authors"],
                "year": paper_meta["year"]
            }

            index.upsert([{
                "id": f"{filename}_{sname}_{i}",
                "values": emb,
                "metadata": meta
            }])
    print(f"Uploaded all sections for {filename}")

# --- Pipeline ---
pdf_dir = os.path.join(os.path.dirname(__file__), "../data/pdfs")
extract_dir = os.path.join(os.path.dirname(__file__), "../data/extracted")
structured_dir = os.path.join(os.path.dirname(__file__), "../data/structured")
os.makedirs(extract_dir, exist_ok=True)
os.makedirs(structured_dir, exist_ok=True)

for fname in os.listdir(pdf_dir):
    if not fname.lower().endswith(".pdf"):
        continue

    meta_info = get_doc_metadata(fname)
    if not meta_info:
        print(f"Skipping {fname} (not found in CSV metadata)")
        continue

    print(f"Processing {fname}...")
    pdf_path = os.path.join(pdf_dir, fname)
    raw_text = extract_pdf_text(pdf_path)

    # Save extracted text
    with open(os.path.join(extract_dir, f"{fname}.txt"), "w", encoding="utf-8") as f:
        f.write(raw_text)

    # Gemini structured JSON
    structured = gemini_extract_sections(raw_text)

    # --- Fallback: Fill missing metadata using regex ---
    if not structured.get("title"):
        structured["title"] = extract_title(raw_text)
    if not structured.get("authors"):
        structured["authors"] = extract_authors(raw_text)
    if not structured.get("year"):
        structured["year"] = extract_year(raw_text)

    # Save improved JSON
    json_path = os.path.join(structured_dir, f"{fname}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2)

    # Upload summarized chunks with improved metadata
    upload_to_pinecone(fname, structured, structured.get("title", meta_info["title"]), meta_info["url"])

print("All PDFs processed, enriched, and uploaded successfully!")
