import os
import re
import pdfplumber
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import tiktoken

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)

index_name = "project-blueprint-index"
encoding = tiktoken.get_encoding("cl100k_base")
if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
index = pc.Index(index_name)

SECTION_NAMES = [
    "Title",
    "Abstract",
    "Introduction",
    "Methods",
    "Results",
    "Discussion",
    "Conclusion",
    "References"
]

def extract_pdf_text(pdf_path):
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)

def extract_sections(text):
    pattern = r'(?i)(' + '|'.join(SECTION_NAMES) + r')\s*[:\n]'
    matches = list(re.finditer(pattern, text))
    starts = [m.start() for m in matches]
    labels = [m.group(1).capitalize() for m in matches]
    positions = starts + [len(text)]
    sections = {}
    if not labels:
        sections["Body"] = text.strip()
        return sections
    for i, label in enumerate(labels):
        start = positions[i]
        end = positions[i+1]
        chunk = text[start:end].strip()
        if chunk and len(chunk) > 25:
            sections[label] = chunk
    # If Title missing but Abstract found, infer Title from start till Abstract
    if "Abstract" in sections and "Title" not in sections:
        title_end = text.find(sections["Abstract"])
        title_text = text[:title_end].strip()
        if title_text:
            sections["Title"] = title_text
    return sections

def summarize_section(text, label):
    prompt = (
        f"Summarize the following {label} section from a scientific document into concise, search-friendly text. For 'Title', just return it:\n\n{text[:2500]}\n"
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def upload_to_pinecone(sections, filename):
    batch = []
    for idx, sname in enumerate(SECTION_NAMES):
        sectext = sections.get(sname)
        if not sectext or len(sectext) < 20:
            continue
        summary = summarize_section(sectext, sname) if sname != "Title" else sectext
        embed_input = summary if sname != "Title" else sectext
        embed = client.embeddings.create(
            model="text-embedding-3-small",
            input=embed_input
        ).data[0].embedding
    
        meta = {
            "filename": filename,
            "section": sname,
            "raw_text": sectext[:500],
        }
        if sname != "Title":
            meta["summary"] = summary
        batch.append({
            "id": f"{filename}_{sname}_{idx}",
            "values": embed,
            "metadata": meta
        })
        print(f"Section processed: {sname}")
    if batch:
        index.upsert(vectors=batch)
        print(f"Uploaded {len(batch)} sections for {filename}")

pdf_dir = "../data"
for fname in os.listdir(pdf_dir):
    if fname.lower().endswith(".pdf"):
        print(f"> Processing: {fname}")
        pdf_path = os.path.join(pdf_dir, fname)
        text = extract_pdf_text(pdf_path)
        text = re.sub(r'\n{2,}', '\n', text)
        sections = extract_sections(text)
        upload_to_pinecone(sections, fname)

print("All PDFs processed and stored in Pinecone by key section.")
