"""
Extract text field from JSON corpus and write to plain text file.

Efficiently processes large JSON files line-by-line without loading entire file
into memory. Handles both JSON array format and newline-delimited JSON (NDJSON).

Usage:
    python scripts/extract_text_from_json.py corpus/data.json corpus/darija_extracted.txt
"""

import json
import sys
from pathlib import Path


def extract_text_streaming(input_path: Path, output_path: Path) -> None:
    """
    Stream-process JSON file and extract 'text' field from each record.
    
    Handles two formats:
    1. JSON array: [{"id": 1, "text": "...", ...}, ...]
    2. NDJSON: {"id": 1, "text": "...", ...}\n{"id": 2, ...}\n...
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output text file (one line per record)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        # Try to detect format by reading first character
        first_char = f_in.read(1)
        f_in.seek(0)
        
        if first_char == '[':
            # JSON array format - need to parse carefully
            print("Detected JSON array format")
            extract_from_json_array(f_in, f_out)
        else:
            # NDJSON format - one JSON object per line
            print("Detected NDJSON format")
            extract_from_ndjson(f_in, f_out)


def extract_from_json_array(f_in, f_out) -> None:
    """Extract from JSON array format: [{"text": "..."}, ...]"""
    # Skip opening bracket
    line = f_in.readline()
    if not line.strip().startswith('['):
        raise ValueError("Expected JSON array starting with '['")
    
    count = 0
    buffer = ""
    in_object = False
    
    for line in f_in:
        buffer += line
        
        # Simple heuristic: if line contains both { and }, it's likely a complete object
        if '{' in line:
            in_object = True
        
        if '}' in line and in_object:
            # Try to parse accumulated buffer
            # Remove trailing comma and closing bracket if present
            obj_str = buffer.strip().rstrip(',').rstrip(']').strip()
            
            if obj_str:
                try:
                    obj = json.loads(obj_str)
                    if 'text' in obj and obj['text'].strip():
                        f_out.write(obj['text'].strip() + '\n')
                        count += 1
                        if count % 10000 == 0:
                            print(f"Processed {count:,} records...")
                except json.JSONDecodeError:
                    # Buffer might be incomplete, continue accumulating
                    continue
            
            buffer = ""
            in_object = False
    
    print(f"✓ Extracted {count:,} text records")


def extract_from_ndjson(f_in, f_out) -> None:
    """Extract from NDJSON format: one JSON object per line."""
    count = 0
    skipped = 0
    
    for line_num, line in enumerate(f_in, 1):
        line = line.strip()
        if not line or line in ['[', ']', ',']:
            continue
        
        # Remove trailing comma if present
        if line.endswith(','):
            line = line[:-1]
        
        try:
            obj = json.loads(line)
            if 'text' in obj and obj['text'].strip():
                f_out.write(obj['text'].strip() + '\n')
                count += 1
                if count % 10000 == 0:
                    print(f"Processed {count:,} records...")
            else:
                skipped += 1
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse line {line_num}: {e}")
            skipped += 1
            continue
    
    print(f"✓ Extracted {count:,} text records")
    if skipped > 0:
        print(f"⚠ Skipped {skipped:,} records (empty text or parse errors)")


def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_text_from_json.py <input.json> <output.txt>")
        print("\nExample:")
        print("  python scripts/extract_text_from_json.py corpus/data.json corpus/darija_extracted.txt")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Size:   {input_path.stat().st_size / 1e6:.1f} MB")
    print()
    
    extract_text_streaming(input_path, output_path)
    
    if output_path.exists():
        output_size = output_path.stat().st_size / 1e6
        print(f"\n✓ Output written: {output_path} ({output_size:.1f} MB)")


if __name__ == "__main__":
    main()
