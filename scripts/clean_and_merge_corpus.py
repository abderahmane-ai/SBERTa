"""
Clean and merge all corpus files into a single clean corpus.

Cleaning steps:
1. Remove lines with <5 tokens
2. Remove lines with >30% non-alphabetic characters (spam, emoji)
3. Remove duplicate and near-duplicate lines

Usage:
    python scripts/clean_and_merge_corpus.py
"""

import re
from pathlib import Path
from collections import defaultdict
from typing import Set, List


# Arabizi digits are linguistic characters, not spam
_ARABIZI_DIGITS = set('23456789')

# Maximum tokens per line (prevents memory issues from malformed data)
MAX_TOKENS = 512


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization."""
    return text.strip().split()


def compute_spam_ratio(text: str) -> float:
    """
    Compute ratio of non-linguistic characters (excluding spaces).
    Treats Arabizi digits (2-9) as valid linguistic content.
    High ratio indicates spam (hhhh, emoji spam, etc.)
    """
    if not text:
        return 1.0
    
    # Count linguistic characters: alphabetic OR Arabizi digits
    linguistic_count = sum(1 for c in text if c.isalpha() or c in _ARABIZI_DIGITS)
    # Count total non-space characters
    total_count = sum(1 for c in text if not c.isspace())
    
    if total_count == 0:
        return 1.0
    
    return 1.0 - (linguistic_count / total_count)


def normalize_for_dedup(text: str) -> str:
    """
    Normalize text for near-duplicate detection.
    - Lowercase
    - Remove extra whitespace
    - Remove punctuation
    """
    # Lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text


def clean_line(line: str, min_tokens: int = 5, max_spam_ratio: float = 0.3) -> str | None:
    """
    Clean a single line. Returns None if line should be filtered out.
    
    Args:
        line: Input line
        min_tokens: Minimum number of tokens required
        max_spam_ratio: Maximum allowed spam ratio
    
    Returns:
        Cleaned line or None if filtered
    """
    line = line.strip()
    
    if not line:
        return None
    
    # Check token count (min and max)
    tokens = tokenize(line)
    if len(tokens) < min_tokens or len(tokens) > MAX_TOKENS:
        return None
    
    # Check spam ratio (Arabizi-aware)
    spam_ratio = compute_spam_ratio(line)
    if spam_ratio > max_spam_ratio:
        return None
    
    return line


def process_file(
    input_path: Path,
    seen_normalized: Set[str],
    min_tokens: int = 5,
    max_spam_ratio: float = 0.3
) -> tuple[List[str], int, int, int]:
    """
    Process a single file and return cleaned lines.
    
    Returns:
        (cleaned_lines, total_lines, filtered_lines, duplicate_lines)
    """
    cleaned_lines = []
    total_lines = 0
    filtered_lines = 0
    duplicate_lines = 0
    
    with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            total_lines += 1
            
            # Clean line
            cleaned = clean_line(line, min_tokens, max_spam_ratio)
            if cleaned is None:
                filtered_lines += 1
                continue
            
            # Check for duplicates
            normalized = normalize_for_dedup(cleaned)
            if normalized in seen_normalized:
                duplicate_lines += 1
                continue
            
            seen_normalized.add(normalized)
            cleaned_lines.append(cleaned)
    
    return cleaned_lines, total_lines, filtered_lines, duplicate_lines


def main():
    # Paths
    old_collection_dir = Path("corpus/archived/old_collection")
    elner_file = Path("corpus/elner/elner.txt")
    output_file = Path("corpus/darija_corpus_clean.txt")
    
    # Collect all .txt files
    old_collection_files = sorted(old_collection_dir.glob("*.txt"))
    
    if not old_collection_files:
        print(f"Error: No .txt files found in {old_collection_dir}")
        return
    
    if not elner_file.exists():
        print(f"Error: ELNER file not found: {elner_file}")
        return
    
    print("="*70)
    print("CORPUS CLEANING AND MERGING")
    print("="*70)
    print(f"Old collection files: {len(old_collection_files)}")
    print(f"ELNER file: {elner_file}")
    print(f"Output: {output_file}")
    print()
    print("Cleaning criteria:")
    print("  - Minimum tokens: 5")
    print(f"  - Maximum tokens: {MAX_TOKENS}")
    print("  - Maximum spam ratio: 30%")
    print("  - Arabizi-aware: digits 2-9 treated as linguistic")
    print("  - Remove duplicates: Yes")
    print()
    print("="*70)
    print()
    
    # Track seen lines for deduplication (across all files)
    seen_normalized: Set[str] = set()
    
    # Statistics
    stats = defaultdict(lambda: {"total": 0, "filtered": 0, "duplicates": 0, "kept": 0})
    all_cleaned_lines = []
    
    # Process old collection files
    print("Processing old collection files...")
    for file_path in old_collection_files:
        print(f"  {file_path.name}...", end=" ", flush=True)
        
        cleaned, total, filtered, duplicates = process_file(
            file_path, seen_normalized
        )
        
        stats[file_path.name]["total"] = total
        stats[file_path.name]["filtered"] = filtered
        stats[file_path.name]["duplicates"] = duplicates
        stats[file_path.name]["kept"] = len(cleaned)
        
        all_cleaned_lines.extend(cleaned)
        
        print(f"{len(cleaned):,} kept (from {total:,})")
    
    print()
    
    # Process ELNER file
    print(f"Processing ELNER file...")
    print(f"  {elner_file.name}...", end=" ", flush=True)
    
    cleaned, total, filtered, duplicates = process_file(
        elner_file, seen_normalized
    )
    
    stats[elner_file.name]["total"] = total
    stats[elner_file.name]["filtered"] = filtered
    stats[elner_file.name]["duplicates"] = duplicates
    stats[elner_file.name]["kept"] = len(cleaned)
    
    all_cleaned_lines.extend(cleaned)
    
    print(f"{len(cleaned):,} kept (from {total:,})")
    print()
    
    # Write output
    print(f"Writing cleaned corpus to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in all_cleaned_lines:
            f.write(line + '\n')
    
    output_size_mb = output_file.stat().st_size / 1e6
    print(f"✓ Written: {len(all_cleaned_lines):,} lines ({output_size_mb:.1f} MB)")
    print()
    
    # Print detailed statistics
    print("="*70)
    print("DETAILED STATISTICS")
    print("="*70)
    print(f"{'File':<45} {'Total':>10} {'Filtered':>10} {'Duplicates':>10} {'Kept':>10}")
    print("-"*70)
    
    total_all = 0
    filtered_all = 0
    duplicates_all = 0
    kept_all = 0
    
    for filename in sorted(stats.keys()):
        s = stats[filename]
        print(f"{filename:<45} {s['total']:>10,} {s['filtered']:>10,} {s['duplicates']:>10,} {s['kept']:>10,}")
        total_all += s['total']
        filtered_all += s['filtered']
        duplicates_all += s['duplicates']
        kept_all += s['kept']
    
    print("-"*70)
    print(f"{'TOTAL':<45} {total_all:>10,} {filtered_all:>10,} {duplicates_all:>10,} {kept_all:>10,}")
    print()
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total input lines:     {total_all:,}")
    print(f"Filtered (<5 tokens or >30% spam): {filtered_all:,} ({filtered_all/total_all*100:.1f}%)")
    print(f"Duplicates removed:    {duplicates_all:,} ({duplicates_all/total_all*100:.1f}%)")
    print(f"Final clean lines:     {kept_all:,} ({kept_all/total_all*100:.1f}%)")
    print()
    print(f"✓ Clean corpus saved: {output_file}")
    print(f"✓ Size: {output_size_mb:.1f} MB")


if __name__ == "__main__":
    main()
