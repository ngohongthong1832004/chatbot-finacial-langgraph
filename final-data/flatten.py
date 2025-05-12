import re
import pandas as pd
from bs4 import BeautifulSoup
import os
import json
from typing import List, Dict, Any, Tuple

def read_html_file(file_path: str) -> str:
    """Read HTML content from a file, trying multiple encodings."""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
    
    # If all encodings fail, try reading in binary mode and then decode
    with open(file_path, 'rb') as file:
        binary_content = file.read()
        # Try to decode with replacement for invalid characters
        return binary_content.decode('utf-8', errors='replace')

def extract_content_from_html(html_content: str) -> Dict[str, Any]:
    """Extract all content from the HTML document."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract document title
    title = soup.find('title')
    title_text = title.get_text(strip=True) if title else "Financial Statement Data Sets"
    
    # Extract all headings and their content
    sections = extract_sections(soup)
    
    # Extract all tables
    tables = extract_tables(soup)
    
    # Extract all numbered lists
    lists = extract_lists(soup)
    
    # Create a complete document structure
    document = {
        "title": title_text,
        "sections": sections,
        "tables": tables,
        "lists": lists
    }
    
    return document

def extract_sections(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Extract all sections with headings and their content."""
    sections = []
    
    # Find all headings
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    
    for i, heading in enumerate(headings):
        heading_text = heading.get_text(strip=True)
        # Handle empty headings
        if not heading_text:
            continue
            
        heading_level = int(heading.name[1])  # h1 -> 1, h2 -> 2, etc.
        
        # Find all content until the next heading of same or higher level
        content = []
        next_elem = heading.find_next()
        
        while next_elem and (not next_elem.name or not (next_elem.name.startswith('h') and 
                                                       int(next_elem.name[1]) <= heading_level)):
            if next_elem.name == 'p':
                try:
                    paragraph_class = next_elem.get('class', [''])[0]
                    if paragraph_class.startswith('MsoCaption'):
                        # Skip captions, as they'll be handled with tables
                        pass
                    else:
                        paragraph_text = next_elem.get_text(strip=True)
                        if paragraph_text:
                            content.append({"type": "paragraph", "text": paragraph_text})
                except (IndexError, AttributeError):
                    # Handle paragraphs without class
                    paragraph_text = next_elem.get_text(strip=True)
                    if paragraph_text:
                        content.append({"type": "paragraph", "text": paragraph_text})
            
            # Check for lists
            elif next_elem.name in ['ol', 'ul']:
                list_items = []
                for li in next_elem.find_all('li'):
                    list_items.append(li.get_text(strip=True))
                
                if list_items:
                    content.append({
                        "type": "list",
                        "items": list_items
                    })
            
            try:
                next_elem = next_elem.find_next()
            except AttributeError:
                break
            
            # Stop if we've reached the next heading of same or higher level
            if next_elem and next_elem.name and next_elem.name.startswith('h') and int(next_elem.name[1]) <= heading_level:
                break
        
        # Extract section number if present
        section_number = None
        section_match = re.search(r'^(\d+(\.\d+)*)', heading_text)
        if section_match:
            section_number = section_match.group(1)
            heading_text = heading_text.replace(section_number, '').strip()
        
        sections.append({
            "level": heading_level,
            "number": section_number,
            "title": heading_text,
            "content": content
        })
    
    return sections

def extract_tables(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Extract all tables from the HTML content."""
    tables = []
    
    # Find all table captions
    captions = soup.find_all('p', class_='MsoCaption')
    
    for caption in captions:
        caption_text = caption.get_text(strip=True)
        
        # Find the table that follows this caption
        table_element = caption.find_next('table')
        if table_element:
            # Parse the table
            table_data = parse_table(table_element)
            
            # Try to determine table type and name from caption
            table_type = "unknown"
            table_name = None
            
            if "Fields in the" in caption_text and "data set" in caption_text:
                table_type = "field_definition"
                match = re.search(r"Fields in the ([A-Z]+) data set", caption_text)
                if match:
                    table_name = match.group(1)
            elif "Data relationships" in caption_text or "relationships" in caption_text.lower():
                table_type = "relationship"
            elif "figure" in caption_text.lower():
                table_type = "figure"
            
            tables.append({
                "caption": caption_text,
                "type": table_type,
                "name": table_name,
                "data": table_data
            })
    
    return tables

def parse_table(table_element) -> Dict[str, Any]:
    """Parse an HTML table into a structured format with better handling for all content."""
    # Get all rows
    rows = table_element.find_all('tr')
    if not rows:
        return {"headers": [], "rows": []}
    
    # Extract headers
    header_row = rows[0]
    headers = []
    for cell in header_row.find_all(['th', 'td']):
        header_text = cell.get_text(strip=True)
        headers.append(header_text)
    
    # Extract data rows
    data_rows = []
    for row in rows[1:]:
        cells = row.find_all(['td', 'th'])
        if cells:
            row_data = []
            for cell in cells:
                cell_text = cell.get_text(strip=True)
                row_data.append(cell_text)
            
            # Make sure row_data has the same length as headers
            while len(row_data) < len(headers):
                row_data.append("")
            
            # Truncate if there are more cells than headers
            if len(row_data) > len(headers):
                row_data = row_data[:len(headers)]
                
            data_rows.append(row_data)
    
    return {
        "headers": headers,
        "rows": data_rows
    }

def extract_lists(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Extract all numbered and bulleted lists."""
    lists = []
    
    # Find all list paragraphs
    list_paragraphs = soup.find_all('p', class_='MsoListParagraph')
    
    current_list = []
    for para in list_paragraphs:
        text = para.get_text(strip=True)
        if text:
            # Check if it's a new list by looking for numbering
            list_marker = None
            list_match = re.match(r'^(\d+\.|•|\*)\s+', text)
            if list_match:
                list_marker = list_match.group(1)
                text = text[len(list_marker):].strip()
            
            current_list.append({
                "marker": list_marker,
                "text": text
            })
    
    if current_list:
        lists.append(current_list)
    
    return lists

def create_flattened_chunks(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create flattened chunks suitable for RAG from the entire document."""
    chunks = []
    
    # Add document title chunk
    chunks.append({
        "id": "document_title",
        "content": document["title"],
        "metadata": {
            "type": "title"
        }
    })
    
    # Add section chunks with their content
    for i, section in enumerate(document["sections"]):
        section_title = section["title"]
        section_number = section["number"] or f"{i+1}"
        
        # Create content from section texts
        content = f"Section {section_number}: {section_title}\n\n"
        
        for item in section["content"]:
            if item["type"] == "paragraph":
                content += item["text"] + "\n\n"
            elif item["type"] == "list":
                for j, list_item in enumerate(item["items"]):
                    content += f"{j+1}. {list_item}\n"
                content += "\n"
        
        chunks.append({
            "id": f"section_{section_number.replace('.', '_')}",
            "content": content.strip(),
            "metadata": {
                "type": "section",
                "section_number": section_number,
                "section_title": section_title,
                "level": section["level"]
            }
        })
    
    # Add table chunks - both complete tables and row-based chunks
    for i, table in enumerate(document["tables"]):
        table_name = table["name"] or f"table_{i+1}"
        
        # 1. Complete table chunk
        table_content = f"Table: {table_name}\nCaption: {table['caption']}\n\n"
        
        # Convert table data to text
        headers = table["data"]["headers"]
        rows = table["data"]["rows"]
        
        # Add headers
        table_content += " | ".join(headers) + "\n"
        table_content += "-" * (len(" | ".join(headers))) + "\n"
        
        # Add rows
        for row in rows:
            table_content += " | ".join(row) + "\n"
        
        chunks.append({
            "id": f"table_{table_name}",
            "content": table_content,
            "metadata": {
                "type": "table",
                "table_name": table_name,
                "table_type": table["type"],
                "caption": table["caption"]
            }
        })
        
        # 2. Field-specific chunks for definition tables
        if table["type"] == "field_definition" and table["name"]:
            field_name_index = headers.index("Field Name") if "Field Name" in headers else 0
            
            for row in rows:
                # Get the field name from the appropriate column
                field_name = row[field_name_index]
                
                # Create a dictionary of field properties
                field_props = {}
                for j, header in enumerate(headers):
                    if j < len(row):
                        field_props[header] = row[j]
                
                # Create field content
                field_content = f"Table: {table_name}\nField: {field_name}\n\n"
                
                # Add all properties
                for header, value in field_props.items():
                    field_content += f"{header}: {value}\n"
                
                chunks.append({
                    "id": f"field_{table_name}_{field_name}",
                    "content": field_content,
                    "metadata": {
                        "type": "field",
                        "table_name": table_name,
                        "field_name": field_name
                    }
                })
        
        # 3. Create relationship chunks for relationship tables
        if table["type"] == "relationship" or "relationship" in table["caption"].lower():
            # Try to identify source, target, and join columns from headers
            source_col = next((h for h in headers if "Dataset" in h), None)
            target_col = next((h for h in headers if "Referenced dataset" in h), None)
            source_fields_col = next((h for h in headers if "referencing" in h.lower()), None)
            target_fields_col = next((h for h in headers if "Referenced columns" in h), None)
            
            if source_col and target_col:
                for row_idx, row in enumerate(rows):
                    # Map header names to row values
                    row_dict = {headers[i]: val for i, val in enumerate(row) if i < len(headers)}
                    
                    source = row_dict.get(source_col, "")
                    target = row_dict.get(target_col, "")
                    source_fields = row_dict.get(source_fields_col, "")
                    target_fields = row_dict.get(target_fields_col, "")
                    
                    if source and target:
                        rel_content = f"Relationship: {source} → {target}\n"
                        if source_fields:
                            rel_content += f"Columns in {source}: {source_fields}\n"
                        if target_fields:
                            rel_content += f"Referenced columns in {target}: {target_fields}\n"
                        
                        chunks.append({
                            "id": f"relationship_{source}_{target}_{row_idx}",
                            "content": rel_content,
                            "metadata": {
                                "type": "relationship",
                                "source_table": source,
                                "target_table": target
                            }
                        })
    
    return chunks

def convert_to_csv_format(document: Dict[str, Any], output_dir: str):
    """Convert document tables to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each table
    for table in document["tables"]:
        table_name = table["name"] or re.sub(r'[^a-zA-Z0-9]', '_', table["caption"])[:30]
        
        # Create DataFrame
        headers = table["data"]["headers"]
        rows = table["data"]["rows"]
        
        df = pd.DataFrame(rows, columns=headers)
        
        # Save to CSV
        csv_path = os.path.join(output_dir, f"{table_name}.csv")
        df.to_csv(csv_path, index=False)
        
        print(f"Saved table '{table_name}' to {csv_path}")

def create_combined_chunks(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create combined chunks for better RAG retrieval."""
    combined_chunks = []
    
    # Get all tables
    tables = {t["name"]: t for t in document["tables"] if t["name"]}
    
    # Create table relationship map
    table_rels = {}
    for table in document["tables"]:
        if table["type"] == "relationship" or "relationship" in table["caption"].lower():
            headers = table["data"]["headers"]
            
            # Find the relevant column indices
            source_idx = next((i for i, h in enumerate(headers) if "Dataset" in h), None)
            target_idx = next((i for i, h in enumerate(headers) if "Referenced dataset" in h), None)
            
            if source_idx is not None and target_idx is not None:
                for row in table["data"]["rows"]:
                    if len(row) > max(source_idx, target_idx):
                        source = row[source_idx]
                        target = row[target_idx]
                        
                        if source not in table_rels:
                            table_rels[source] = []
                        table_rels[source].append(target)
    
    # Process each table definition and create combined chunks
    for table_name, table in tables.items():
        # Find the related section
        related_section = None
        for section in document["sections"]:
            if table_name in section["title"]:
                related_section = section
                break
        
        # Create combined content
        combined_content = f"Table: {table_name}\n\n"
        
        # Add section content if available
        if related_section:
            for item in related_section["content"]:
                if item["type"] == "paragraph":
                    combined_content += item["text"] + "\n\n"
        
        # Add table definition
        combined_content += f"Fields in the {table_name} data set:\n\n"
        
        headers = table["data"]["headers"]
        field_name_idx = headers.index("Field Name") if "Field Name" in headers else 0
        desc_idx = next((i for i, h in enumerate(headers) if "Description" in h), 1)
        
        for row in table["data"]["rows"]:
            if len(row) > max(field_name_idx, desc_idx):
                field_name = row[field_name_idx]
                description = row[desc_idx] if desc_idx < len(row) else ""
                combined_content += f"- {field_name}: {description}\n"
        
        # Add relationship information
        if table_name in table_rels:
            combined_content += f"\nThe {table_name} table references the following tables:\n"
            for target in table_rels[table_name]:
                combined_content += f"- {target}\n"
        
        # Add tables that reference this table
        referenced_by = []
        for source, targets in table_rels.items():
            if table_name in targets:
                referenced_by.append(source)
        
        if referenced_by:
            combined_content += f"\nThe {table_name} table is referenced by the following tables:\n"
            for source in referenced_by:
                combined_content += f"- {source}\n"
        
        combined_chunks.append({
            "id": f"combined_{table_name}",
            "content": combined_content,
            "metadata": {
                "type": "combined",
                "table_name": table_name
            }
        })
    
    return combined_chunks

def main(input_file: str, output_dir: str = "flattened_sec_data"):
    """Main function to process the HTML file and generate flattened data."""
    print(f"Reading HTML file: {input_file}")
    html_content = read_html_file(input_file)
    
    print(f"Extracting content from HTML...")
    document = extract_content_from_html(html_content)
    
    print(f"Found {len(document['sections'])} sections and {len(document['tables'])} tables")
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save complete document structure
    with open(os.path.join(output_dir, "document_structure.json"), 'w', encoding='utf-8') as f:
        json.dump(document, f, indent=2)
    
    # Create and save RAG-friendly chunks
    chunks = create_flattened_chunks(document)
    with open(os.path.join(output_dir, "rag_chunks.jsonl"), 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')
    
    # Create and save combined chunks
    combined_chunks = create_combined_chunks(document)
    with open(os.path.join(output_dir, "combined_chunks.jsonl"), 'w', encoding='utf-8') as f:
        for chunk in combined_chunks:
            f.write(json.dumps(chunk) + '\n')
    
    # Convert tables to CSV format
    convert_to_csv_format(document, os.path.join(output_dir, "csv_tables"))
    
    print(f"Done! Output saved to {output_dir}/")
    print(f"Created {len(chunks)} standard chunks and {len(combined_chunks)} combined chunks for RAG")

if __name__ == "__main__":
    import sys
    
    # Use command line argument if provided, otherwise default to "readme.htm"
    input_file = sys.argv[1] if len(sys.argv) > 1 else "readme.htm"
    
    main(input_file)