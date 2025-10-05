import os, json, pdfplumber
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI
from pinecone import Pinecone
import tiktoken

# --- Load keys ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# --- Initialize clients ---
genai.configure(api_key=GOOGLE_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "nasa-bio-embeddings"
if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(name=index_name, dimension=1536, metric="cosine")
index = pc.Index(index_name)

encoding = tiktoken.get_encoding("cl100k_base")

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
def upload_to_pinecone(filename, metadata):
    sections = metadata.get("sections", {})
    paper_meta = {
        "paper_id": metadata.get("paper_id", ""),
        "title": metadata.get("title", ""),
        "authors": metadata.get("authors", []),
        "year": metadata.get("year", ""),
        "doi": metadata.get("doi", "")
    }

    for sname, stext in sections.items():
        if not stext.strip():
            continue
        summary = summarize_section(stext, sname)
        chunks = chunk_text(summary)
        for i, chunk in enumerate(chunks):
            emb = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk
            ).data[0].embedding
            meta = {**paper_meta, "section": sname, "text_preview": chunk[:200]}
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

    print(f"Processing {fname}...")
    pdf_path = os.path.join(pdf_dir, fname)
    raw_text = extract_pdf_text(pdf_path)

    # Save extracted text
    with open(os.path.join(extract_dir, f"{fname}.txt"), "w", encoding="utf-8") as f:
        f.write(raw_text)

    # Gemini for structured JSON
    structured = gemini_extract_sections(raw_text)

    # Save JSON
    json_path = os.path.join(structured_dir, f"{fname}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2)

    # Upload summarized chunks
    upload_to_pinecone(fname, structured)

print("All PDFs processed successfully!")
