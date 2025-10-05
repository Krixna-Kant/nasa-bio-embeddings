"""
Relation extraction module using SciSpacy for biomedical entity extraction
and LLM for relation identification. Processes PDF JSON files and prepares
triples for Neo4j database.
"""

import json
import os
import re
import time
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import spacy
import nltk
from nltk.tokenize import sent_tokenize
import openai
from openai import OpenAI


@dataclass
class Entity:
    """Represents a biomedical entity extracted by SciSpacy"""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 0.0


@dataclass
class Relation:
    """Represents a relation between two entities"""
    subject: Entity
    predicate: str
    object: Entity
    confidence: float
    evidence: str


@dataclass
class Triple:
    """Represents a Neo4j triple with metadata"""
    subject: str
    subject_type: str
    predicate: str
    object: str
    object_type: str
    paper_id: str
    section: str
    evidence: str
    confidence: float


class SciSpacyEntityExtractor:
    """Extracts biomedical entities using SciSpacy models"""
    
    def __init__(self, models: List[str] = None):
        if models is None:
            # Use both models for comprehensive entity extraction
            models = ["en_core_sci_sm", "en_ner_bc5cdr_md"]
        
        self.nlp_models = {}
        for model_name in models:
            try:
                self.nlp_models[model_name] = spacy.load(model_name)
                print(f"Loaded model: {model_name}")
            except OSError:
                print(f"Warning: SciSpacy model '{model_name}' not found. Skipping.")
        
        if not self.nlp_models:
            raise ValueError("No SciSpacy models could be loaded. Please install at least one model.")
        
        # Download NLTK data if not present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract biomedical entities from text using all available SciSpacy models"""
        all_entities = []
        seen_entities = set()  # To avoid duplicates
        
        for model_name, nlp in self.nlp_models.items():
            doc = nlp(text)
            
            for ent in doc.ents:
                # Create a unique key for the entity (text + position)
                entity_key = (ent.text.lower(), ent.start_char, ent.end_char)
                
                if entity_key not in seen_entities:
                    # Map generic ENTITY labels to more specific types where possible
                    label = ent.label_
                    if label == "ENTITY":
                        # Comprehensive heuristics for NASA space life sciences research
                        text_lower = ent.text.lower()
                        
                        # Core biological entities
                        if any(word in text_lower for word in ['protein', 'enzyme', 'antibody', 'peptide', 'amino acid', 'polypeptide']):
                            label = "PROTEIN"
                        elif any(word in text_lower for word in ['bacteria', 'fungus', 'microbe', 'organism', 'archaea', 'virus', 'yeast', 'escherichia', 'bacillus']):
                            label = "ORGANISM"
                        elif any(word in text_lower for word in ['cell', 'neuron', 'tissue', 'organ', 'membrane', 'cytoplasm', 'nucleus', 'mitochondria']):
                            label = "CELL_TYPE"
                        elif any(word in text_lower for word in ['gene', 'dna', 'rna', 'genome', 'chromosome', 'nucleotide', 'mrna', 'sequence', 'allele']):
                            label = "GENETIC_MATERIAL"
                        elif any(word in text_lower for word in ['drug', 'compound', 'molecule', 'chemical', 'ion', 'salt', 'reagent', 'substrate']):
                            label = "CHEMICAL"
                            
                        # Space environment entities
                        elif any(word in text_lower for word in ['radiation', 'cosmic ray', 'solar', 'uv', 'gamma', 'ionizing', 'spe', 'ger', 'hze']):
                            label = "SPACE_RADIATION"
                        elif any(word in text_lower for word in ['microgravity', 'gravity', 'weightless', 'zero-g', 'hypergravity', '1g', 'partial gravity']):
                            label = "GRAVITATIONAL_CONDITION"
                        elif any(word in text_lower for word in ['mars', 'moon', 'lunar', 'planetary', 'asteroid', 'comet', 'europa', 'titan', 'iss location']):
                            label = "CELESTIAL_BODY"
                        elif any(word in text_lower for word in ['space station', 'spacecraft', 'iss', 'habitat', 'capsule', 'shuttle', 'dragon', 'soyuz']):
                            label = "SPACE_VEHICLE"
                        elif any(word in text_lower for word in ['atmosphere', 'vacuum', 'pressure', 'oxygen', 'co2', 'nitrogen', 'partial pressure', 'ppco2']):
                            label = "ATMOSPHERIC_CONDITION"
                            
                        # NASA-specific life support and biotechnology
                        elif any(word in text_lower for word in ['life support', 'blss', 'bioregenerative', 'recycling', 'scrubber', 'ecls', 'environmental control']):
                            label = "LIFE_SUPPORT_SYSTEM"
                        elif any(word in text_lower for word in ['food', 'nutrition', 'vitamin', 'calorie', 'diet', 'meal', 'food system', 'crop', 'plant growth']):
                            label = "NUTRITION"
                        elif any(word in text_lower for word in ['exercise', 'fitness', 'muscle', 'bone', 'countermeasure', 'ared', 'treadmill', 'colp']):
                            label = "PHYSIOLOGICAL_COUNTERMEASURE"
                        elif any(word in text_lower for word in ['sleep', 'circadian', 'rhythm', 'light', 'darkness', 'melatonin', 'photoperiod']):
                            label = "CIRCADIAN_FACTOR"
                            
                        # Research methodologies and NASA mission elements
                        elif any(word in text_lower for word in ['experiment', 'protocol', 'procedure', 'method', 'technique', 'assay', 'analysis', 'measurement']):
                            label = "RESEARCH_METHOD"
                        elif any(word in text_lower for word in ['mission', 'eva', 'spacewalk', 'expedition', 'flight', 'increment', 'sts', 'expedition']):
                            label = "MISSION_ACTIVITY"
                        elif any(word in text_lower for word in ['astronaut', 'crew', 'human', 'subject', 'participant', 'crewmember', 'flight crew']):
                            label = "HUMAN_SUBJECT"
                        elif any(word in text_lower for word in ['rodent', 'mouse', 'rat', 'drosophila', 'c. elegans', 'zebrafish', 'arabidopsis']):
                            label = "MODEL_ORGANISM"
                            
                        # Space medicine and health
                        elif any(word in text_lower for word in ['bone loss', 'muscle atrophy', 'cardiovascular', 'neurovestibular', 'renal', 'immune system']):
                            label = "PHYSIOLOGICAL_SYSTEM"
                        elif any(word in text_lower for word in ['medication', 'pharmaceutical', 'therapeutic', 'treatment', 'intervention', 'therapy']):
                            label = "MEDICAL_INTERVENTION"
                        elif any(word in text_lower for word in ['biomarker', 'indicator', 'metric', 'parameter', 'endpoint', 'outcome measure']):
                            label = "BIOMARKER"
                            
                        # Environmental and ecological systems
                        elif any(word in text_lower for word in ['ecosystem', 'biosphere', 'environment', 'habitat', 'niche', 'closed ecosystem']):
                            label = "ECOSYSTEM"
                        elif any(word in text_lower for word in ['resource', 'water', 'mineral', 'soil', 'regolith', 'consumables', 'supplies']):
                            label = "RESOURCE"
                        elif any(word in text_lower for word in ['temperature', 'thermal', 'heat', 'cold', 'freezing', 'heating', 'cooling']):
                            label = "THERMAL_CONDITION"
                            
                        # NASA hardware and technology
                        elif any(word in text_lower for word in ['centrifuge', 'incubator', 'microscope', 'spectrometer', 'analyzer', 'sensor', 'camera']):
                            label = "RESEARCH_EQUIPMENT"
                        elif any(word in text_lower for word in ['facility', 'rack', 'glovebox', 'freezer', 'stowage', 'laboratory', 'module']):
                            label = "SPACE_FACILITY"
                        elif any(word in text_lower for word in ['sample', 'specimen', 'tissue sample', 'blood sample', 'urine sample', 'biological sample']):
                            label = "BIOLOGICAL_SAMPLE"
                            
                        # NASA mission phases and operations
                        elif any(word in text_lower for word in ['preflight', 'inflight', 'postflight', 'launch', 'landing', 'recovery', 'return']):
                            label = "MISSION_PHASE"
                        elif any(word in text_lower for word in ['payload', 'cargo', 'manifest', 'resupply', 'logistics', 'transport']):
                            label = "MISSION_LOGISTICS"
                            
                        # Data and information systems
                        elif any(word in text_lower for word in ['database', 'repository', 'archive', 'dataset', 'data collection', 'osdr']):
                            label = "DATA_SYSTEM"
                        elif any(word in text_lower for word in ['grant', 'funding', 'task book', 'investigation', 'study', 'research']):
                            label = "RESEARCH_PROGRAM"
                            
                        # Fallback for other biological entities
                        else:
                            label = "BIOENTITY"
                    
                    entity = Entity(
                        text=ent.text,
                        label=label,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=0.8
                    )
                    all_entities.append(entity)
                    seen_entities.add(entity_key)
        
        return all_entities


class LLMRelationExtractor:
    """Extracts relations between entities using OpenAI's API with error handling"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            # Try to get from environment
            self.client = OpenAI()
        
        self.model = model
        self.max_tokens = 4000  # Token limit guard for API calls
        self.max_retries = 3  # Number of retries for failed API calls
        self.retry_delay = 1  # Initial delay between retries (seconds)
        
    def _count_tokens(self, text: str) -> int:
        """Rough estimate of token count (approximately 4 chars per token)"""
        return len(text) // 4
    
    def _create_relation_prompt(self, sentence: str, entities: List[Entity]) -> str:
        """Create a prompt for relation extraction"""
        entity_list = []
        for i, entity in enumerate(entities):
            entity_list.append(f"{i+1}. {entity.text} ({entity.label})")
        
        entities_str = "\n".join(entity_list)
        
        prompt = f"""
Analyze the following scientific sentence from NASA space life sciences research and identify relationships between the entities listed below.

Sentence: "{sentence}"

Entities:
{entities_str}

Please identify meaningful biological, medical, or space science relationships. Focus on NASA research patterns including:

**Biological/Medical Relations:**
- inhibits, activates, regulates, modulates, enhances, suppresses, upregulates, downregulates
- causes, prevents, treats, induces, triggers, correlates_with, associated_with
- interacts_with, binds_to, metabolizes, synthesizes, produces, secretes
- located_in, part_of, contains, composed_of, derived_from
- affects, influences, controls, determines, alters, changes

**Space Flight Effects & Adaptations:**
- exposed_to, irradiated_by, affected_by (radiation, microgravity effects)
- protected_by, shielded_from, mitigated_by, counteracted_by
- adapted_to, responds_to, tolerates, survives_in, resistant_to
- degrades_in, deteriorates_under, stable_in, preserved_in
- increased_during, decreased_during, unchanged_in (flight conditions)

**Mission & Research Operations:**
- required_for, essential_for, necessary_for, critical_for (life support, mission success)
- operates_in, functions_in, deployed_on, used_in, conducted_on
- measured_during, observed_in, studied_in, monitored_throughout
- collected_from, sampled_during, analyzed_post, returned_from
- compared_to, normalized_to, relative_to (ground controls, preflight)

**NASA-Specific Relations:**
- countermeasure_for, prevents_loss_of, maintains_function_of
- facility_houses, equipment_measures, system_provides
- investigation_studies, experiment_examines, protocol_specifies
- payload_contains, mission_transports, crew_operates

For each relationship, provide:
1. Subject entity (by number)
2. Relationship type using the above terms or similar NASA research vocabulary
3. Object entity (by number) 
4. Confidence score (0.0-1.0) - be conservative but realistic

Format your response as JSON:
{{
  "relations": [
    {{
      "subject": 1,
      "predicate": "relationship_type", 
      "object": 2,
      "confidence": 0.8
    }}
  ]
}}

Only return relationships explicitly stated or strongly implied in the sentence. Focus on scientifically meaningful connections relevant to space life sciences research.
"""
        return prompt
    
    def _call_api_with_retry(self, messages: List[Dict], max_tokens: int = 500) -> str:
        """Call OpenAI API with retry logic and rate limiting"""
        for attempt in range(self.max_retries):
            try:
                # Add a small delay to avoid rate limiting
                if attempt > 0:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Retrying API call in {delay} seconds... (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(delay)
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.1
                )
                
                # Successfully got response
                return response.choices[0].message.content
                
            except Exception as api_error:
                print(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {api_error}")
                
                # If this was the last attempt, raise the error
                if attempt == self.max_retries - 1:
                    raise api_error
                    
                # For rate limit errors, wait longer
                if "rate_limit" in str(api_error).lower():
                    time.sleep(10)
        
        return ""  # Should never reach here
    
    def extract_relations(self, sentence: str, entities: List[Entity]) -> List[Relation]:
        """Extract relations between entities in a sentence using LLM"""
        if len(entities) < 2:
            return []
        
        prompt = self._create_relation_prompt(sentence, entities)
        
        # Check token limit
        if self._count_tokens(prompt) > self.max_tokens:
            print(f"Warning: Prompt exceeds token limit, truncating entities list")
            # Truncate entities if prompt is too long
            entities = entities[:5]  # Keep only first 5 entities
            prompt = self._create_relation_prompt(sentence, entities)
        
        try:
            # Use retry mechanism for API call
            messages = [
                {"role": "system", "content": "You are a biomedical relation extraction expert. Extract only valid scientific relationships."},
                {"role": "user", "content": prompt}
            ]
            
            response_text = self._call_api_with_retry(messages, max_tokens=500)
            
            # Debug: Print response for troubleshooting
            if not response_text or response_text.strip() == "":
                print(f"Warning: Empty response from LLM for sentence: {sentence[:50]}...")
                return []
            
            # Clean response text (remove potential markdown formatting)
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Try to parse JSON
            try:
                response_data = json.loads(response_text)
            except json.JSONDecodeError as json_error:
                print(f"JSON parsing error: {json_error}")
                print(f"Raw response: '{response_text[:200]}...'")
                print(f"Sentence that caused error: {sentence[:100]}...")
                
                # Try to extract JSON from malformed response
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        response_data = json.loads(json_match.group())
                        print("Successfully recovered JSON from malformed response")
                    except json.JSONDecodeError:
                        print("Could not recover JSON, skipping this sentence")
                        return []
                else:
                    print("No JSON structure found in response, skipping")
                    return []
            
            relations = []
            for rel_data in response_data.get("relations", []):
                try:
                    subject_idx = rel_data["subject"] - 1  # Convert to 0-based index
                    object_idx = rel_data["object"] - 1
                    
                    if 0 <= subject_idx < len(entities) and 0 <= object_idx < len(entities):
                        relation = Relation(
                            subject=entities[subject_idx],
                            predicate=rel_data["predicate"],
                            object=entities[object_idx],
                            confidence=rel_data["confidence"],
                            evidence=sentence
                        )
                        relations.append(relation)
                except (KeyError, IndexError, ValueError) as e:
                    print(f"Error parsing relation: {e}")
                    print(f"Relation data: {rel_data}")
                    continue
            
            return relations
            
        except Exception as e:
            print(f"Error in LLM relation extraction: {e}")
            print(f"Error type: {type(e).__name__}")
            print(f"Sentence: {sentence[:100]}...")
            print(f"Number of entities: {len(entities)}")
            return []


class Neo4jTripleBuilder:
    """Builds Neo4j-ready triples from relations"""
    
    def __init__(self):
        pass
    
    def relation_to_triple(self, relation: Relation, paper_id: str, section: str) -> Triple:
        """Convert a relation to a Neo4j triple"""
        return Triple(
            subject=relation.subject.text,
            subject_type=relation.subject.label,
            predicate=relation.predicate,
            object=relation.object.text,
            object_type=relation.object.label,
            paper_id=paper_id,
            section=section,
            evidence=relation.evidence,
            confidence=relation.confidence
        )
    
    def triples_to_cypher(self, triples: List[Triple]) -> List[str]:
        """Convert triples to Cypher CREATE statements"""
        cypher_statements = []
        
        for triple in triples:
            # Escape quotes in text
            subject_clean = triple.subject.replace("'", "\\'").replace('"', '\\"')
            object_clean = triple.object.replace("'", "\\'").replace('"', '\\"')
            evidence_clean = triple.evidence.replace("'", "\\'").replace('"', '\\"')
            predicate_clean = triple.predicate.replace(" ", "_").upper()
            
            cypher = f"""
CREATE (s:{triple.subject_type} {{name: '{subject_clean}', paper_id: '{triple.paper_id}'}})
CREATE (o:{triple.object_type} {{name: '{object_clean}', paper_id: '{triple.paper_id}'}})
CREATE (s)-[r:{predicate_clean} {{
    confidence: {triple.confidence}, 
    section: '{triple.section}', 
    evidence: '{evidence_clean}',
    paper_id: '{triple.paper_id}'
}}]->(o)
"""
            cypher_statements.append(cypher.strip())
        
        return cypher_statements


class PDFRelationProcessor:
    """Main processor for extracting relations from PDF JSON files - optimized for NASA OSDR scale"""
    
    def __init__(self, openai_api_key: str = None):
        self.entity_extractor = SciSpacyEntityExtractor()
        self.relation_extractor = LLMRelationExtractor(api_key=openai_api_key)
        self.triple_builder = Neo4jTripleBuilder()
        
        # Extended target sections for comprehensive NASA research processing
        self.target_sections = [
            "Results", "Conclusion", "Conclusions", "Discussion", 
            "Abstract", "Summary", "Key Findings", "Significance",
            "Methods", "Materials and Methods", "Experimental Design",
            "Background", "Introduction"  # These often contain important context
        ]
        
        # Processing statistics for large-scale operations
        self.stats = {
            'files_processed': 0,
            'total_sentences': 0,
            'total_entities': 0,
            'total_relations': 0,
            'errors': 0
        }
    
    def process_sentence(self, sentence: str, paper_id: str, section: str) -> List[Triple]:
        """Process a single sentence and return triples"""
        # Extract entities
        entities = self.entity_extractor.extract_entities(sentence)
        
        if len(entities) < 2:
            return []
        
        # Extract relations
        relations = self.relation_extractor.extract_relations(sentence, entities)
        
        # Convert to triples
        triples = []
        for relation in relations:
            triple = self.triple_builder.relation_to_triple(relation, paper_id, section)
            triples.append(triple)
        
        return triples
    
    def process_json_file(self, json_file_path: str, save_individual: bool = True) -> List[Triple]:
        """Process a single PDF JSON file and extract all triples"""
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        paper_id = data["filename"].replace('.pdf', '')
        original_filename = data["filename"]
        all_triples = []
        
        # Process each section
        for section_data in data["sections"]:
            section_name = section_data["section"]
            
            # Only process Results and Conclusions sections
            if section_name in self.target_sections:
                text = section_data["text"]
                
                # Split into sentences
                sentences = sent_tokenize(text)
                
                print(f"Processing {len(sentences)} sentences from {section_name} section of {paper_id}")
                
                # Process each sentence
                for i, sentence in enumerate(sentences):
                    if len(sentence.strip()) < 20:  # Skip very short sentences
                        continue
                    
                    print(f"  Sentence {i+1}/{len(sentences)}: {sentence[:100]}...")
                    
                    try:
                        triples = self.process_sentence(sentence, paper_id, section_name)
                        all_triples.extend(triples)
                        
                        if triples:
                            print(f"    Found {len(triples)} relation(s):")
                            for j, triple in enumerate(triples):
                                print(f"      {j+1}. {triple.subject} --{triple.predicate}--> {triple.object} (confidence: {triple.confidence})")
                        else:
                            print(f"    No relations found")
                        
                    except Exception as e:
                        print(f"    Error processing sentence: {e}")
                        continue
        
        # Save individual results if requested
        if save_individual:
            self.save_individual_results(json_file_path, all_triples, original_filename)
        
        return all_triples
    
    def process_pdfs_folder(self, pdfs_folder_path: str, batch_size: int = 50) -> Dict[str, List[Triple]]:
        """Process all JSON files in the pdfs folder with batch processing for large datasets"""
        results = {}
        
        # Find all JSON files (excluding relations files)
        json_files = [f for f in os.listdir(pdfs_folder_path) 
                     if f.endswith('.json') and not f.endswith('_relations.json')]
        
        total_files = len(json_files)
        print(f"Found {total_files} JSON files to process from NASA repositories")
        print(f"Processing in batches of {batch_size} files")
        
        # Process in batches for memory management
        for batch_start in range(0, total_files, batch_size):
            batch_end = min(batch_start + batch_size, total_files)
            batch_files = json_files[batch_start:batch_end]
            
            print(f"\n=== BATCH {batch_start//batch_size + 1}: Files {batch_start+1}-{batch_end} ===")
            
            for i, json_file in enumerate(batch_files):
                file_path = os.path.join(pdfs_folder_path, json_file)
                file_number = batch_start + i + 1
                
                print(f"\n[{file_number}/{total_files}] Processing {json_file}...")
                
                try:
                    triples = self.process_json_file(file_path, save_individual=True)
                    results[json_file] = triples
                    
                    # Update statistics
                    self.stats['files_processed'] += 1
                    self.stats['total_relations'] += len(triples)
                    
                    print(f"✅ Extracted {len(triples)} triples from {json_file}")
                    
                    # Print sample relations for verification
                    if triples:
                        print(f"   Sample relations:")
                        for j, triple in enumerate(triples[:2]):  # Show first 2
                            print(f"     • {triple.subject} --{triple.predicate}--> {triple.object}")
                            print(f"       (confidence: {triple.confidence:.2f}, section: {triple.section})")
                    
                    # Progress summary every 10 files
                    if file_number % 10 == 0:
                        self._print_progress_summary(file_number, total_files)
                        
                except Exception as e:
                    print(f"❌ Error processing {json_file}: {e}")
                    self.stats['errors'] += 1
                    results[json_file] = []
            
            # Memory cleanup between batches
            if batch_end < total_files:
                print(f"\nCompleted batch {batch_start//batch_size + 1}. Preparing next batch...")
        
        return results
    
    def _print_progress_summary(self, current: int, total: int):
        """Print progress summary for large-scale processing"""
        percentage = (current / total) * 100
        print(f"\n📊 PROGRESS SUMMARY ({current}/{total} - {percentage:.1f}%)")
        print(f"   Files processed: {self.stats['files_processed']}")
        print(f"   Total relations found: {self.stats['total_relations']}")
        print(f"   Errors encountered: {self.stats['errors']}")
        if current > 0:
            avg_relations = self.stats['total_relations'] / self.stats['files_processed']
            print(f"   Average relations per file: {avg_relations:.1f}")
        print("─" * 50)
    
    def save_individual_results(self, json_file_path: str, triples: List[Triple], original_filename: str):
        """Save extraction results for a single file with enhanced NASA OSDR metadata"""
        # Get the base name without extension
        base_name = os.path.splitext(os.path.basename(json_file_path))[0]
        
        # Create output filename with _relations suffix
        output_file = os.path.join(os.path.dirname(json_file_path), f"{base_name}_relations.json")
        
        # Convert triples to serializable format with enhanced metadata
        serializable_triples = []
        entity_types_found = set()
        relation_types_found = set()
        
        for triple in triples:
            serializable_triples.append({
                "subject": triple.subject,
                "subject_type": triple.subject_type,
                "predicate": triple.predicate,
                "object": triple.object,
                "object_type": triple.object_type,
                "paper_id": triple.paper_id,
                "section": triple.section,
                "evidence": triple.evidence,
                "confidence": triple.confidence
            })
            
            # Collect statistics
            entity_types_found.add(triple.subject_type)
            entity_types_found.add(triple.object_type)
            relation_types_found.add(triple.predicate)
        
        # Create the output structure with enhanced NASA OSDR metadata
        output_data = {
            "filename": original_filename,
            "paper_id": base_name,
            "extraction_metadata": {
                "extraction_date": "2025-10-05",
                "extractor_version": "NASA-BioMind-v1.0",
                "target_sections": self.target_sections,
                "total_relations": len(serializable_triples),
                "entity_types_found": sorted(list(entity_types_found)),
                "relation_types_found": sorted(list(relation_types_found)),
                "sources": [
                    "NASA Open Science Data Repository (OSDR)",
                    "NASA Space Life Sciences Library", 
                    "NASA Task Book"
                ]
            },
            "relations": serializable_triples
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Relations saved to {output_file}")
        print(f"  - {len(serializable_triples)} relations")
        print(f"  - {len(entity_types_found)} entity types: {', '.join(sorted(list(entity_types_found))[:5])}{'...' if len(entity_types_found) > 5 else ''}")
        print(f"  - {len(relation_types_found)} relation types: {', '.join(sorted(list(relation_types_found))[:3])}{'...' if len(relation_types_found) > 3 else ''}")
    
    def generate_cypher_statements(self, results: Dict[str, List[Triple]], output_file: str = "neo4j_statements.cypher"):
        """Generate Cypher statements for Neo4j import"""
        all_triples = []
        for triples in results.values():
            all_triples.extend(triples)
        
        cypher_statements = self.triple_builder.triples_to_cypher(all_triples)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("// Neo4j Cypher statements for relation import\n\n")
            for statement in cypher_statements:
                f.write(statement + "\n\n")
        
        print(f"Cypher statements saved to {output_file}")


def main():
    """Main function to run NASA space life sciences relation extraction"""
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    print("🚀 NASA BioMind - Space Life Sciences Relation Extraction")
    print("=" * 60)
    print("Sources: NASA OSDR, Space Life Sciences Library, Task Book")
    print("Optimized for processing 600+ research papers")
    print("=" * 60)
    
    # Get OpenAI API key
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in the .env file")
        return
    
    # Initialize processor
    print("🔧 Initializing NASA space life sciences processor...")
    processor = PDFRelationProcessor(openai_api_key=openai_api_key)
    
    # Process PDFs folder
    pdfs_folder = "pdfs"  # Relative to this script's location
    if not os.path.exists(pdfs_folder):
        print(f"❌ Error: PDFs folder '{pdfs_folder}' not found")
        print("Please ensure your NASA research papers are in the 'pdfs' directory")
        return
    
    print(f"📁 Processing papers from: {os.path.abspath(pdfs_folder)}")
    
    # Run extraction with batch processing for large datasets
    results = processor.process_pdfs_folder(pdfs_folder, batch_size=25)
    
    # Generate cypher statements from all results
    print("\n🔗 Generating Neo4j Cypher statements...")
    processor.generate_cypher_statements(results, "nasa_osdr_neo4j_import.cypher")
    
    # Comprehensive summary for NASA dataset
    total_triples = sum(len(triples) for triples in results.values())
    successful_files = sum(1 for triples in results.values() if len(triples) > 0)
    
    print(f"\n{'='*60}")
    print(f"🎯 NASA SPACE LIFE SCIENCES EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"📊 PROCESSING STATISTICS:")
    print(f"   Total files processed: {len(results)}")
    print(f"   Successful extractions: {successful_files}")
    print(f"   Failed extractions: {processor.stats['errors']}")
    print(f"   Total relations extracted: {total_triples}")
    print(f"   Average relations per successful file: {total_triples/successful_files if successful_files > 0 else 0:.1f}")
    
    print(f"\n📈 TOP PERFORMING FILES:")
    # Sort by number of relations found
    sorted_results = sorted(results.items(), key=lambda x: len(x[1]), reverse=True)
    for i, (file_name, triples) in enumerate(sorted_results[:5]):
        if triples:
            print(f"   {i+1}. {file_name}: {len(triples)} relations")
            # Show most common entity types in this file
            entity_types = {}
            for triple in triples:
                entity_types[triple.subject_type] = entity_types.get(triple.subject_type, 0) + 1
                entity_types[triple.object_type] = entity_types.get(triple.object_type, 0) + 1
            top_entities = sorted(entity_types.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"      Top entities: {', '.join([f'{t}({c})' for t, c in top_entities])}")
    
    print(f"\n🗄️  DATABASE INTEGRATION:")
    print(f"   Neo4j import file: nasa_osdr_neo4j_import.cypher")
    print(f"   Individual relation files: {successful_files} *_relations.json files")
    print(f"   Ready for knowledge graph visualization!")
    
    print(f"\n🔬 RESEARCH COVERAGE:")
    all_entity_types = set()
    all_relation_types = set()
    for triples in results.values():
        for triple in triples:
            all_entity_types.add(triple.subject_type)
            all_entity_types.add(triple.object_type)
            all_relation_types.add(triple.predicate)
    
    print(f"   Entity types discovered: {len(all_entity_types)}")
    print(f"   Relationship types: {len(all_relation_types)}")
    print(f"   Sample entities: {', '.join(sorted(list(all_entity_types))[:8])}...")
    print(f"   Sample relations: {', '.join(sorted(list(all_relation_types))[:6])}...")
    
    print(f"\n✅ Ready for NASA BioMind knowledge graph analysis!")
    print(f"   Start the web interface to explore the extracted relationships")
    print(f"   All data is now compatible with OSDR, Task Book, and Library sources")


if __name__ == "__main__":
    main()
