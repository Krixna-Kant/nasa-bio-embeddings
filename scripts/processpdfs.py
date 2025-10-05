import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import PyPDF2
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
            
            return text.strip()
    
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
        return ""

def chunk_text(text: str, max_chunk_size: int = 10000) -> List[str]:
    """
    Split text into smaller chunks to fit within LLM context limits.
    
    Args:
        text (str): Text to chunk
        max_chunk_size (int): Maximum characters per chunk
        
    Returns:
        List[str]: List of text chunks
    """
    # Split by paragraphs first to maintain coherence
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed the limit, start a new chunk
        if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
    
    # Add the last chunk if it has content
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def process_text_with_llm(text: str, filename: str, attempt: int = 1) -> Dict[str, Any]:
    """
    Process extracted text with LLM. First tries to process the full text,
    and only chunks recursively when it fails due to context length.
    
    Args:
        text (str): Raw text from PDF
        filename (str): Name of the PDF file for reference
        attempt (int): Current attempt number for logging
        
    Returns:
        Dict[str, Any]: Dictionary containing sections, authors, year, title, and summary
    """
    prompt = """You are an NLP scientist. The following text is raw paragraphs from a scientific paper. 

Extract the following information:
1. Authors: List of author names (as strings)
2. Year: Publication year (as integer)
3. Title: Paper title (as string)
4. Summary: A comprehensive 2-3 sentence summary of the entire paper
5. Sections: Group text into standard scientific sections

For sections, organize into ALL of these standard scientific sections: Abstract, Introduction, Methods, Results, Discussion, Conclusion.

IMPORTANT: You MUST return exactly 6 sections in this exact order. If a section is not explicitly present in the text, try to construct it from available information or use "N/A - This section was not found in the original document" if truly missing.

Return JSON in this exact format:
{
  "authors": ["Author 1", "Author 2", "..."],
  "year": 2024,
  "title": "Paper Title",
  "summary": "Comprehensive summary of the paper...",
  "sections": [
    {"section":"Abstract","text":"..."},
    {"section":"Introduction","text":"..."},
    {"section":"Methods","text":"..."},
    {"section":"Results","text":"..."},
    {"section":"Discussion","text":"..."},
    {"section":"Conclusion","text":"..."}
  ]
}

Text to process:
"""
    
    try:
        logger.info(f"Processing {filename} (attempt {attempt}, text length: {len(text)} chars)")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert NLP scientist specializing in scientific paper analysis. Return only valid JSON without any additional text or formatting. Extract metadata (authors, year, title, summary) and organize into exactly 6 sections in the specified order."
                },
                {
                    "role": "user", 
                    "content": prompt + text
                }
            ],
            max_tokens=4000,
            temperature=0.1
        )
        
        # Extract the response content
        response_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            result = json.loads(response_text)
            # Ensure we have the required structure
            if not all(key in result for key in ['authors', 'year', 'title', 'summary', 'sections']):
                logger.warning(f"Missing required fields in response for {filename}. Creating default structure...")
                result = get_default_result("Missing required fields in LLM response")
            elif len(result.get('sections', [])) != 6:
                logger.warning(f"Expected 6 sections, got {len(result.get('sections', []))} for {filename}. Fixing...")
                result['sections'] = ensure_all_sections(result.get('sections', []))
            logger.info(f"Successfully processed {filename} in full on attempt {attempt}")
            return result
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response for {filename}")
            return get_default_result("Failed to parse LLM response as JSON")
    
    except Exception as e:
        error_str = str(e)
        # Check if the error is due to context length
        if "context_length_exceeded" in error_str or "maximum context length" in error_str:
            logger.warning(f"Context length exceeded for {filename} on attempt {attempt}, splitting in half")
            return process_text_in_chunks_recursive(text, filename, attempt)
        else:
            logger.error(f"Error processing text with LLM for {filename}: {error_str}")
            return get_default_result(f"LLM processing failed: {error_str}")

def process_text_in_chunks_recursive(text: str, filename: str, attempt: int = 1) -> Dict[str, Any]:
    """
    Recursively process text by splitting in half when context length is exceeded.
    
    Args:
        text (str): Raw text from PDF
        filename (str): Name of the PDF file for reference
        attempt (int): Current attempt number for logging
        
    Returns:
        Dict[str, Any]: Dictionary containing sections, authors, year, title, and summary
    """
    # Split text in half at paragraph boundaries
    paragraphs = text.split('\n\n')
    mid_point = len(paragraphs) // 2
    
    first_half = '\n\n'.join(paragraphs[:mid_point])
    second_half = '\n\n'.join(paragraphs[mid_point:])
    
    logger.info(f"Splitting {filename} into 2 parts (attempt {attempt})")
    
    # Process each half recursively
    first_results = process_text_with_llm(first_half, f"{filename}_part1", attempt + 1)
    second_results = process_text_with_llm(second_half, f"{filename}_part2", attempt + 1)
    
    # Combine the results using LLM
    combined_results = combine_chunked_results_with_llm(first_results, second_results, filename)
    
    return combined_results

def combine_chunked_results_with_llm(first_results: Dict[str, Any], second_results: Dict[str, Any], filename: str) -> Dict[str, Any]:
    """
    Combine results from multiple chunks using LLM to ensure coherent sections.
    
    Args:
        first_results (Dict[str, Any]): Results from first chunk
        second_results (Dict[str, Any]): Results from second chunk
        filename (str): Name of the PDF file for reference
        
    Returns:
        Dict[str, Any]: Combined and cleaned result with metadata and sections
    """
    # Convert results to text for LLM processing
    first_text = json.dumps(first_results, indent=2)
    second_text = json.dumps(second_results, indent=2)
    
    prompt = f"""You are an NLP scientist. I have processed a scientific paper in two parts and got the following extractions:

Part 1:
{first_text}

Part 2:
{second_text}

Please combine these into a single, coherent result. For metadata (authors, year, title, summary), use the most complete and accurate information from either part. For sections, merge content that belongs to the same section, maintain the logical flow, and ensure all text is preserved.

IMPORTANT: You MUST return exactly 6 sections in this exact order: Abstract, Introduction, Methods, Results, Discussion, Conclusion. If any section is missing from the input, try to construct it from available information or use "N/A - This section was not found in the original document".

Return JSON in this exact format:
{{
  "authors": ["Author 1", "Author 2", "..."],
  "year": 2024,
  "title": "Paper Title",
  "summary": "Comprehensive summary of the paper...",
  "sections": [
    {{"section":"Abstract","text":"..."}},
    {{"section":"Introduction","text":"..."}},
    {{"section":"Methods","text":"..."}},
    {{"section":"Results","text":"..."}},
    {{"section":"Discussion","text":"..."}},
    {{"section":"Conclusion","text":"..."}}
  ]
}}
"""
    
    try:
        logger.info(f"Combining chunked results for {filename}")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert NLP scientist specializing in scientific paper analysis. Combine and organize sections logically. Return only valid JSON without any additional text or formatting. Extract metadata (authors, year, title, summary) and organize into exactly 6 sections in the specified order."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=4000,
            temperature=0.1
        )
        
        response_text = response.choices[0].message.content.strip()
        
        try:
            combined_result = json.loads(response_text)
            # Ensure we have the required structure
            if not all(key in combined_result for key in ['authors', 'year', 'title', 'summary', 'sections']):
                logger.warning(f"Missing required fields in combined response for {filename}. Using manual combination...")
                return manual_combine_results(first_results, second_results)
            elif len(combined_result.get('sections', [])) != 6:
                logger.warning(f"Expected 6 sections after combining, got {len(combined_result.get('sections', []))} for {filename}. Fixing...")
                combined_result['sections'] = ensure_all_sections(combined_result.get('sections', []))
            logger.info(f"Successfully combined sections for {filename}")
            return combined_result
        except json.JSONDecodeError:
            logger.error(f"Failed to parse combined JSON response for {filename}")
            # Fallback: manually combine sections
            return manual_combine_results(first_results, second_results)
    
    except Exception as e:
        logger.error(f"Error combining sections for {filename}: {str(e)}")
        # Fallback: manually combine sections
        return manual_combine_results(first_results, second_results)

def ensure_all_sections(sections: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Ensure all 6 standard sections are present in the correct order.
    
    Args:
        sections (List[Dict[str, str]]): Input sections
        
    Returns:
        List[Dict[str, str]]: Sections with all 6 standard sections present
    """
    standard_sections = ["Abstract", "Introduction", "Methods", "Results", "Discussion", "Conclusion"]
    sections_dict = {}
    
    # Extract existing sections
    for section in sections:
        section_name = section.get("section", "")
        section_text = section.get("text", "")
        
        # Map to standard section names
        if section_name in standard_sections:
            sections_dict[section_name] = section_text
        else:
            # Try to match non-standard names
            section_lower = section_name.lower()
            if "abstract" in section_lower:
                sections_dict["Abstract"] = section_text
            elif "introduction" in section_lower or "intro" in section_lower:
                sections_dict["Introduction"] = section_text
            elif "method" in section_lower or "methodology" in section_lower:
                sections_dict["Methods"] = section_text
            elif "result" in section_lower or "finding" in section_lower:
                sections_dict["Results"] = section_text
            elif "discussion" in section_lower or "analysis" in section_lower:
                sections_dict["Discussion"] = section_text
            elif "conclusion" in section_lower or "summary" in section_lower:
                sections_dict["Conclusion"] = section_text
    
    # Create final list with all sections
    result = []
    for section_name in standard_sections:
        section_text = sections_dict.get(section_name, "N/A - This section was not found in the original document")
        result.append({"section": section_name, "text": section_text})
    
    return result

def get_default_sections(error_message: str) -> List[Dict[str, str]]:
    """
    Return default sections structure when processing fails.
    
    Args:
        error_message (str): Error message to include
        
    Returns:
        List[Dict[str, str]]: Default sections with error message
    """
    standard_sections = ["Abstract", "Introduction", "Methods", "Results", "Discussion", "Conclusion"]
    return [{"section": section, "text": f"Error: {error_message}"} for section in standard_sections]

def get_default_result(error_message: str) -> Dict[str, Any]:
    """
    Return default result structure when processing fails.
    
    Args:
        error_message (str): Error message to include
        
    Returns:
        Dict[str, Any]: Default result with error message
    """
    return {
        "authors": [f"Error: {error_message}"],
        "year": None,
        "title": f"Error: {error_message}",
        "summary": f"Error: {error_message}",
        "sections": get_default_sections(error_message)
    }

def manual_combine_sections(first_results: List[Dict[str, str]], second_results: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Manually combine sections as a fallback when LLM combination fails.
    Ensures all standard sections are present.
    
    Args:
        first_results (List[Dict[str, str]]): Results from first chunk
        second_results (List[Dict[str, str]]): Results from second chunk
        
    Returns:
        List[Dict[str, str]]: Manually combined sections with all standard sections
    """
    # Standard sections in order
    standard_sections = ["Abstract", "Introduction", "Methods", "Results", "Discussion", "Conclusion"]
    all_sections = {section: "" for section in standard_sections}
    
    # Combine sections from both results
    for results in [first_results, second_results]:
        for section_data in results:
            section_name = section_data.get("section", "Unknown")
            section_text = section_data.get("text", "")
            
            # Map to standard section names if close match
            if section_name in all_sections:
                if all_sections[section_name]:
                    all_sections[section_name] += "\n\n" + section_text
                else:
                    all_sections[section_name] = section_text
            else:
                # Try to find the best match for non-standard section names
                section_lower = section_name.lower()
                if "abstract" in section_lower:
                    target_section = "Abstract"
                elif "introduction" in section_lower or "intro" in section_lower:
                    target_section = "Introduction"
                elif "method" in section_lower or "methodology" in section_lower:
                    target_section = "Methods"
                elif "result" in section_lower or "finding" in section_lower:
                    target_section = "Results"
                elif "discussion" in section_lower or "analysis" in section_lower:
                    target_section = "Discussion"
                elif "conclusion" in section_lower or "summary" in section_lower:
                    target_section = "Conclusion"
                else:
                    # Put unmatched content in Discussion as a fallback
                    target_section = "Discussion"
                
                if all_sections[target_section]:
                    all_sections[target_section] += "\n\n" + section_text
                else:
                    all_sections[target_section] = section_text
    
    # Ensure all sections have content, use N/A if empty
    result = []
    for section_name in standard_sections:
        section_text = all_sections[section_name].strip()
        if not section_text:
            section_text = "N/A - This section was not found in the original document"
        result.append({"section": section_name, "text": section_text})
    
    return result

def manual_combine_results(first_results: Dict[str, Any], second_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Manually combine results as a fallback when LLM combination fails.
    
    Args:
        first_results (Dict[str, Any]): Results from first chunk
        second_results (Dict[str, Any]): Results from second chunk
        
    Returns:
        Dict[str, Any]: Manually combined result with metadata and sections
    """
    # Combine metadata - prefer non-empty values, first_results takes priority
    authors = first_results.get('authors', [])
    if not authors or (len(authors) == 1 and 'Error:' in authors[0]):
        authors = second_results.get('authors', [])
    
    year = first_results.get('year')
    if not year:
        year = second_results.get('year')
    
    title = first_results.get('title', '')
    if not title or 'Error:' in title:
        title = second_results.get('title', '')
    
    summary = first_results.get('summary', '')
    if not summary or 'Error:' in summary:
        summary = second_results.get('summary', '')
    
    # Combine sections using existing logic
    combined_sections = manual_combine_sections(
        first_results.get('sections', []), 
        second_results.get('sections', [])
    )
    
    return {
        "authors": authors if authors else ["Unknown"],
        "year": year,
        "title": title if title else "Unknown Title",
        "summary": summary if summary else "No summary available",
        "sections": combined_sections
    }

def save_output(filename: str, result: Dict[str, Any], pdf_directory: str) -> str:
    """
    Save the processed result to a JSON file in the same directory as the PDF.
    
    Args:
        filename (str): Name of the PDF file
        result (Dict[str, Any]): Processed result with metadata and sections
        pdf_directory (str): Directory containing PDF files
        
    Returns:
        str: Path to the saved JSON file, or empty string if failed
    """
    try:
        json_filename = os.path.splitext(filename)[0] + ".json"
        json_path = os.path.join(pdf_directory, json_filename)
        
        json_output = {
            "filename": filename,
            "authors": result.get('authors', []),
            "year": result.get('year'),
            "title": result.get('title', ''),
            "summary": result.get('summary', ''),
            "sections": result.get('sections', []),
            "total_sections": len(result.get('sections', [])),
            "processing_timestamp": datetime.now().isoformat()
        }
        
        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_output, json_file, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved JSON output to: {json_path}")
        return json_path
        
    except Exception as e:
        logger.error(f"Failed to save JSON file for {filename}: {str(e)}")
        return ""

def get_pdf_files(pdf_directory: str) -> List[str]:
    """
    Get all PDF files from the specified directory.
    
    Args:
        pdf_directory (str): Path to directory containing PDFs
        
    Returns:
        List[str]: List of PDF file paths
    """
    pdf_files = []
    pdf_dir = Path(pdf_directory)
    
    if pdf_dir.exists() and pdf_dir.is_dir():
        for file in pdf_dir.glob("*.pdf"):
            pdf_files.append(str(file))
    
    return pdf_files

def process_all_pdfs(pdf_directory: str = "pdfs") -> Dict[str, Any]:
    """
    Process all PDFs in the specified directory.
    
    Args:
        pdf_directory (str): Directory containing PDF files
        
    Returns:
        Dict[str, Any]: Results for all processed PDFs
    """
    results = {
        "processed_files": [],
        "errors": [],
        "summary": {
            "total_files": 0,
            "successful": 0,
            "failed": 0
        }
    }
    
    # Get list of PDF files
    pdf_files = get_pdf_files(pdf_directory)
    results["summary"]["total_files"] = len(pdf_files)
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_directory}")
        return results
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        logger.info(f"Processing: {filename}")
        
        try:
            # Extract text from PDF
            extracted_text = extract_text_from_pdf(pdf_path)
            
            if not extracted_text:
                results["errors"].append({
                    "file": filename,
                    "error": "No text could be extracted from PDF"
                })
                continue
            
            # Process with LLM
            result = process_text_with_llm(extracted_text, filename)
            
            # Save JSON output to file
            json_path = save_output(filename, result, pdf_directory)
            
            # Add to results
            results["processed_files"].append({
                "filename": filename,
                "authors": result.get('authors', []),
                "year": result.get('year'),
                "title": result.get('title', ''),
                "summary": result.get('summary', ''),
                "sections": result.get('sections', []),
                "total_sections": len(result.get('sections', [])),
                "json_output_path": json_path if json_path else "Failed to save"
            })
            
            results["summary"]["successful"] += 1
            logger.info(f"Successfully processed: {filename}")
            
        except Exception as e:
            error_msg = f"Failed to process {filename}: {str(e)}"
            logger.error(error_msg)
            results["errors"].append({
                "file": filename,
                "error": str(e)
            })
            results["summary"]["failed"] += 1
    
    return results

def main():
    """
    Main function to process PDFs and output JSON results.
    """
    logger.info("Starting PDF processing...")
    
    # Check if OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        logger.error("OPENAI_API_KEY environment variable not set")
        print(json.dumps({
            "error": "OPENAI_API_KEY environment variable not set",
            "message": "Please set your OpenAI API key in the .env file"
        }, indent=2))
        return
    
    # Process all PDFs
    results = process_all_pdfs()
    
    # Output results as JSON
    print(json.dumps(results, indent=2))
    
    # Log summary
    summary = results["summary"]
    logger.info(f"Processing complete. Success: {summary['successful']}, Failed: {summary['failed']}, Total: {summary['total_files']}")

if __name__ == "__main__":
    main()
