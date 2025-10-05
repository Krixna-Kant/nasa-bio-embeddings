#!/usr/bin/env python3
"""
Neo4j import utility for extracted relations
"""

import json
import os
from typing import List, Dict
from neo4j import GraphDatabase

class Neo4jImporter:
    """Import extracted relations into        # Ask user which import method to use
        print("\nSelect import method:")
        print("1. Import from consolidated extracted_relations.json (legacy)")
        print("2. Import from individual *_relations.json files in pdfs folder")
        
        choice = input("Enter choice (1 or 2, default=2): ").strip()
        
        if choice == '1':
            # Legacy import method
            print("\nImporting from extracted_relations.json...")
            importer.import_relations_file()
        else:
            # New import method
            pdfs_folder = input("\nEnter pdfs folder path (default='pdfs'): ").strip() or 'pdfs'
            print(f"\nImporting from *_relations.json files in '{pdfs_folder}' folder...")
            importer.import_all_relations_files(pdfs_folder)o4j database"""
    
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
    
    def close(self):
        self.driver.close()
    
    def clear_existing_data(self):
        """Clear existing biomedical relation data"""
        with self.driver.session(database=self.database) as session:
            # Delete relations and nodes with paper_id property
            session.run("MATCH (n) WHERE n.paper_id IS NOT NULL DETACH DELETE n")
            print("Cleared existing biomedical data")
    
    def create_constraints(self):
        """Create database constraints and indexes"""
        with self.driver.session(database=self.database) as session:
            try:
                # Create constraints for entity uniqueness
                session.run("CREATE CONSTRAINT entity_name_paper IF NOT EXISTS FOR (n:BIOENTITY) REQUIRE (n.name, n.paper_id) IS UNIQUE")
                session.run("CREATE CONSTRAINT organism_name_paper IF NOT EXISTS FOR (n:ORGANISM) REQUIRE (n.name, n.paper_id) IS UNIQUE")
                session.run("CREATE CONSTRAINT chemical_name_paper IF NOT EXISTS FOR (n:CHEMICAL) REQUIRE (n.name, n.paper_id) IS UNIQUE")
                session.run("CREATE CONSTRAINT disease_name_paper IF NOT EXISTS FOR (n:DISEASE) REQUIRE (n.name, n.paper_id) IS UNIQUE")
                
                # Create indexes for fast lookups
                session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (n:BIOENTITY) ON (n.name)")
                session.run("CREATE INDEX bioentity_paper_id IF NOT EXISTS FOR (n:BIOENTITY) ON (n.paper_id)")
                session.run("CREATE INDEX organism_paper_id IF NOT EXISTS FOR (n:ORGANISM) ON (n.paper_id)")
                session.run("CREATE INDEX chemical_paper_id IF NOT EXISTS FOR (n:CHEMICAL) ON (n.paper_id)")
                session.run("CREATE INDEX disease_paper_id IF NOT EXISTS FOR (n:DISEASE) ON (n.paper_id)")
                
                print("Created constraints and indexes")
            except Exception as e:
                print(f"Note: Some constraints/indexes may already exist: {e}")
    
    def import_triple(self, triple: Dict) -> bool:
        """Import a single triple into Neo4j"""
        try:
            with self.driver.session(database=self.database) as session:
                # Clean predicate name for relationship type
                predicate_clean = triple['predicate'].replace(" ", "_").replace("-", "_").upper()
                
                # Clean text values
                subject_clean = triple['subject'].replace("'", "\\'").replace('"', '\\"')
                object_clean = triple['object'].replace("'", "\\'").replace('"', '\\"')
                evidence_clean = triple['evidence'].replace("'", "\\'").replace('"', '\\"')
                
                # Create Cypher query using MERGE to avoid duplicates
                query = f"""
                MERGE (s:{triple['subject_type']} {{name: $subject_name, paper_id: $paper_id}})
                MERGE (o:{triple['object_type']} {{name: $object_name, paper_id: $paper_id}})
                MERGE (s)-[r:{predicate_clean}]->(o)
                SET r.confidence = $confidence,
                    r.section = $section,
                    r.evidence = $evidence,
                    r.paper_id = $paper_id
                """
                
                session.run(query, {
                    "subject_name": subject_clean,
                    "object_name": object_clean,
                    "paper_id": triple['paper_id'],
                    "confidence": triple['confidence'],
                    "section": triple['section'],
                    "evidence": evidence_clean
                })
                
                return True
                
        except Exception as e:
            print(f"Error importing triple: {e}")
            return False
    
    def delete_paper_relations(self, paper_id: str):
        """Delete all existing relations for a specific paper"""
        with self.driver.session(database=self.database) as session:
            # Delete relationships for this paper
            result = session.run("""
                MATCH ()-[r]->() 
                WHERE r.paper_id = $paper_id 
                DELETE r
                RETURN count(r) as deleted_rels
            """, {"paper_id": paper_id})
            
            deleted_rels = result.single()["deleted_rels"]
            
            # Delete orphaned nodes for this paper
            result = session.run("""
                MATCH (n) 
                WHERE n.paper_id = $paper_id 
                AND NOT (n)-[]-() 
                DELETE n
                RETURN count(n) as deleted_nodes
            """, {"paper_id": paper_id})
            
            deleted_nodes = result.single()["deleted_nodes"]
            
            if deleted_rels > 0 or deleted_nodes > 0:
                print(f"  Deleted {deleted_rels} relationships and {deleted_nodes} orphaned nodes for paper {paper_id}")
    
    def import_individual_relations_file(self, relations_file: str) -> bool:
        """Import relations from a single *_relations.json file"""
        try:
            with open(relations_file, 'r') as f:
                data = json.load(f)
            
            # Extract paper_id from filename in the JSON
            filename = data.get('filename', '')
            if not filename:
                print(f"Warning: No filename found in {relations_file}")
                return False
            
            paper_id = filename.replace('.pdf', '')
            relations = data.get('relations', [])
            
            print(f"Processing {os.path.basename(relations_file)} (paper: {paper_id})")
            
            # Delete existing relations for this paper
            self.delete_paper_relations(paper_id)
            
            # Import new relations
            successful_imports = 0
            for triple in relations:
                if self.import_triple(triple):
                    successful_imports += 1
            
            print(f"  Imported {successful_imports}/{len(relations)} triples")
            return True
            
        except FileNotFoundError:
            print(f"Error: {relations_file} not found")
            return False
        except Exception as e:
            print(f"Error importing {relations_file}: {e}")
            return False
    
    def find_relations_files(self, pdfs_folder: str = "pdfs") -> List[str]:
        """Find all *_relations.json files in the specified folder"""
        relations_files = []
        
        if not os.path.exists(pdfs_folder):
            print(f"Error: Folder '{pdfs_folder}' not found")
            return relations_files
        
        for filename in os.listdir(pdfs_folder):
            if filename.endswith('_relations.json'):
                full_path = os.path.join(pdfs_folder, filename)
                relations_files.append(full_path)
        
        return sorted(relations_files)
    
    def import_all_relations_files(self, pdfs_folder: str = "pdfs"):
        """Import all *_relations.json files from the specified folder"""
        relations_files = self.find_relations_files(pdfs_folder)
        
        if not relations_files:
            print(f"No *_relations.json files found in '{pdfs_folder}' folder")
            return
        
        print(f"Found {len(relations_files)} relation files to import:")
        for f in relations_files:
            print(f"  - {os.path.basename(f)}")
        
        successful_files = 0
        total_relations = 0
        
        for relations_file in relations_files:
            if self.import_individual_relations_file(relations_file):
                successful_files += 1
                # Count relations in this file
                try:
                    with open(relations_file, 'r') as f:
                        data = json.load(f)
                        total_relations += len(data.get('relations', []))
                except:
                    pass
        
        print(f"\n=== BATCH IMPORT COMPLETE ===")
        print(f"Files processed: {successful_files}/{len(relations_files)}")
        print(f"Total relations imported: {total_relations}")
    
    def get_statistics(self):
        """Get statistics about imported data"""
        with self.driver.session(database=self.database) as session:
            # Count nodes by type
            node_counts = session.run("""
                MATCH (n) WHERE n.paper_id IS NOT NULL
                RETURN labels(n)[0] as label, count(n) as count
                ORDER BY count DESC
            """).data()
            
            # Count relationships
            rel_count = session.run("""
                MATCH ()-[r]->() WHERE r.paper_id IS NOT NULL
                RETURN count(r) as count
            """).single()['count']
            
            # Count papers
            paper_count = session.run("""
                MATCH (n) WHERE n.paper_id IS NOT NULL
                RETURN count(DISTINCT n.paper_id) as count
            """).single()['count']
            
            print(f"\n=== NEO4J DATABASE STATISTICS ===")
            print(f"Papers: {paper_count}")
            print(f"Total relationships: {rel_count}")
            print(f"Nodes by type:")
            for item in node_counts:
                print(f"  {item['label']}: {item['count']}")
    
    def sample_queries(self):
        """Run some sample queries to demonstrate the data"""
        print(f"\n=== SAMPLE QUERIES ===")
        
        with self.driver.session(database=self.database) as session:
            # Query 1: Find all organisms and what they do
            print("\\n1. Organisms and their actions:")
            results = session.run("""
                MATCH (o:ORGANISM)-[r]->(target)
                RETURN o.name as organism, type(r) as action, target.name as target
                LIMIT 5
            """).data()
            
            for result in results:
                print(f"   {result['organism']} --{result['action']}--> {result['target']}")
            
            # Query 2: Find high-confidence relations
            print("\\n2. High-confidence relations (>= 0.9):")
            results = session.run("""
                MATCH (s)-[r]->(o)
                WHERE r.confidence >= 0.9 AND r.paper_id IS NOT NULL
                RETURN s.name as subject, type(r) as predicate, o.name as object, r.confidence as conf
                LIMIT 5
            """).data()
            
            for result in results:
                print(f"   {result['subject']} --{result['predicate']}--> {result['object']} ({result['conf']})")
            
            # Query 3: Relations by paper
            print("\\n3. Relations by paper:")
            results = session.run("""
                MATCH (s)-[r]->(o)
                WHERE r.paper_id IS NOT NULL
                RETURN r.paper_id as paper, count(r) as relations
                ORDER BY relations DESC
            """).data()
            
            for result in results:
                print(f"   {result['paper']}: {result['relations']} relations")


def main():
    """Main function to import relations into Neo4j"""
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Get Neo4j credentials
    neo4j_uri = os.getenv('NEO4J_URI')
    neo4j_username = os.getenv('NEO4J_USERNAME')
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    neo4j_database = os.getenv('NEO4J_DATABASE', 'neo4j')
    
    if not all([neo4j_uri, neo4j_username, neo4j_password]):
        print("Error: Neo4j credentials not found in environment variables")
        print("Please set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD in .env file")
        return
    
    print("=== NEO4J IMPORT UTILITY ===")
    print(f"Connecting to: {neo4j_uri}")
    print(f"Database: {neo4j_database}")
    
    # Initialize importer
    importer = None
    try:
        importer = Neo4jImporter(neo4j_uri, neo4j_username, neo4j_password, neo4j_database)
        
        # Setup database
        print("\\nSetting up database...")
        importer.create_constraints()
        
        # Ask if user wants to clear existing data
        response = input("\\nClear existing biomedical data? (y/N): ").strip().lower()
        if response == 'y':
            importer.clear_existing_data()
        
        # Import from individual *_relations.json files
        pdfs_folder = input("\\nEnter pdfs folder path (default='pdfs'): ").strip() or 'pdfs'
        print(f"\\nImporting from *_relations.json files in '{pdfs_folder}' folder...")
        importer.import_all_relations_files(pdfs_folder)
        
        # Show statistics
        importer.get_statistics()
        
        # Run sample queries
        importer.sample_queries()
        
        print("\\n=== IMPORT COMPLETE ===")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if importer:
            importer.close()


if __name__ == "__main__":
    main()