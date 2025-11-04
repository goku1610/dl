#!/usr/bin/env python3
import argparse
import csv
import sys
import time
from typing import Optional

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"


def call_ollama(prompt: str, model: str = MODEL_NAME, temperature: float = 0.3, max_retries: int = 3, timeout_s: int = 60) -> str:
	"""Call local Ollama HTTP API to generate a single completion.

	Returns the generated text (stripped). Raises on repeated failure.
	"""
	last_err: Optional[Exception] = None
	for attempt in range(1, max_retries + 1):
		try:
			resp = requests.post(
				OLLAMA_URL,
				json={
					"model": model,
					"prompt": prompt,
					"stream": False,
					"options": {
						"temperature": temperature,
					}
				},
				timeout=timeout_s,
			)
			resp.raise_for_status()
			data = resp.json()
			text = data.get("response", "").strip()
			if text:
				return text
			# Empty response; treat as retryable
			last_err = RuntimeError("Empty response from Ollama")
		except Exception as e:  # noqa: BLE001
			last_err = e
			# Exponential backoff with jitter
			sleep_s = min(2 ** attempt, 8) + (0.1 * attempt)
			time.sleep(sleep_s)
	# Exhausted retries
	raise RuntimeError(f"Ollama request failed after {max_retries} attempts: {last_err}")


def build_prompt(class_label: str, attributes: str) -> str:
	"""Construct a concise prompt to produce a short, semantically specific caption.

	Uses real examples from the dataset to guide the model.
	"""
	# Parse attributes to understand what information is available
	attr_dict = {}
	for attr in attributes.split(';'):
		if ':' in attr:
			key, value = attr.split(':', 1)
			attr_dict[key.strip()] = value.strip()

	# Present clear instructions with real examples
	return (
		"You create short, natural product captions based on the object class and attributes.\n"
		"Rules:\n"
		"- Output a concise, grammatically correct noun phrase.\n"
		"- Use lowercase only.\n"
		"- Include condition, size, color, material, features, and pattern when available.\n"
		"- Describe visual state when relevant (e.g., 'flipped over', 'with closed lid', 'with plain pattern').\n"
		"- Use 'with' for secondary parts or features (e.g., 'with brown lid', 'with plain pattern').\n"
		"- Keep it specific but brief (5-12 words typically).\n"
		"- Make it semantically correct and natural.\n\n"
		"Examples:\n"
		"Object class: electronic_accessories_earphones\n"
		"Attributes: color:black;material:unknown;condition:unknown;size:small\n"
		"Caption: black bluetooth earphones with closed lid\n\n"
		"Object class: personal_care_soap_bar\n"
		"Attributes: color:green;material:unknown;condition:new;size:medium\n"
		"Caption: a new green colored soap bar\n\n"
		"Object class: stationary_notebook\n"
		"Attributes: color:black;material:plastic;condition:deformed;size:medium\n"
		"Caption: deformed medium black plastic notebook flipped over\n\n"
		"Object class: wrist_watch\n"
		"Attributes: color:pink;material:rubber;pattern:plain;condition:new;features:smart\n"
		"Caption: a pink rubber smart wrist watch with plain pattern in new condition\n\n"
		"Object class: comb\n"
		"Attributes: color:pink;material:plastic;pattern:plain;condition:new;features:general\n"
		"Caption: a new pink general comb of plastic material with a plain pattern\n\n"
		f"Object class: {class_label}\n"
		f"Attributes: {attributes}\n\n"
		"Caption:"
	)


def write_csv(write_path: str, rows: list[dict], fieldnames: list[str]) -> None:
	"""Helper function to write CSV rows."""
	with open(write_path, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def fill_captions(in_csv: str, out_csv: Optional[str] = None, overwrite: bool = False, limit: Optional[int] = None) -> None:
	"""Fill missing captions using local Ollama and write to output CSV.

	- If overwrite is True, write back to in_csv.
	- If out_csv is provided and overwrite is False, write to out_csv.
	- If neither, default to overwriting in_csv.
	- Only updates rows where caption is empty/blank.
	- All rows with the same class_label will get the same caption.
	- CSV is updated incrementally after each caption generation.
	- limit: optional cap on number of unique class_labels to generate (for testing).
	"""
	read_path = in_csv
	write_path = in_csv if overwrite or not out_csv else out_csv

	rows = []
	class_label_captions: dict[str, str] = {}  # Maps class_label -> caption
	class_label_attributes: dict[str, str] = {}  # Maps class_label -> representative attributes

	# First pass: collect rows and existing captions per class_label
	with open(read_path, newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		fieldnames = reader.fieldnames
		if not fieldnames:
			raise RuntimeError("CSV has no header")
		# Ensure required columns exist
		for col in ("class_label", "attributes", "caption"):
			if col not in fieldnames:
				raise RuntimeError(f"CSV missing required column: {col}")

		for row in reader:
			class_label = (row.get("class_label") or "").strip()
			caption = (row.get("caption") or "").strip()
			attributes = (row.get("attributes") or "").strip()

			# Store the first attributes we see for each class_label
			if class_label and class_label not in class_label_attributes:
				class_label_attributes[class_label] = attributes

			# If this row has a caption, use it for all rows with this class_label
			if caption and class_label:
				class_label_captions[class_label] = caption

			rows.append(row)

	# Apply existing captions to all rows with the same class_label
	for row in rows:
		class_label = (row.get("class_label") or "").strip()
		current_caption = (row.get("caption") or "").strip()

		if class_label in class_label_captions and not current_caption:
			row["caption"] = class_label_captions[class_label]

	# Write initial state (preserves existing captions and applies them)
	write_csv(write_path, rows, fieldnames)

	# Second pass: generate captions for class_labels that need them, updating CSV incrementally
	generated_count = 0
	skipped_count = 0
	updated_count = 0

	for class_label, attributes in class_label_attributes.items():
		# Skip if we already have a caption for this class_label
		if class_label in class_label_captions:
			continue

		# Check limit
		if limit is not None and generated_count >= limit:
			break

		# Generate caption for this class_label
		prompt = build_prompt(class_label, attributes)
		try:
			gen = call_ollama(prompt)
			# Clean: enforce single-line noun phrase, strip trailing punctuation and lowercase
			gen = gen.replace("\n", " ").strip().rstrip(".,;:!?").lower()
			class_label_captions[class_label] = gen
			generated_count += 1

			# Apply caption to all rows with this class_label
			rows_updated_for_this_label = 0
			for row in rows:
				row_class_label = (row.get("class_label") or "").strip()
				current_caption = (row.get("caption") or "").strip()

				if row_class_label == class_label and not current_caption:
					row["caption"] = gen
					rows_updated_for_this_label += 1
					updated_count += 1

			# Write CSV immediately after each caption generation
			write_csv(write_path, rows, fieldnames)
			print(f"Generated caption for '{class_label}': '{gen}' (updated {rows_updated_for_this_label} rows)")

		except Exception as e:  # noqa: BLE001
			print(f"Warn: generation failed for class_label '{class_label}': {e}", file=sys.stderr)

	# Count skipped rows (rows that already had captions)
	for row in rows:
		current_caption = (row.get("caption") or "").strip()
		if current_caption:
			skipped_count += 1

	print(f"\nCompleted: Generated {generated_count} unique captions; updated {updated_count} rows; skipped {skipped_count} existing; wrote: {write_path}")


def clear_captions(in_csv: str, out_csv: Optional[str] = None, overwrite: bool = False) -> None:
	"""Clear all captions from the CSV file."""
	read_path = in_csv
	write_path = in_csv if overwrite or not out_csv else out_csv

	rows = []
	with open(read_path, newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		fieldnames = reader.fieldnames
		if not fieldnames:
			raise RuntimeError("CSV has no header")
		
		if "caption" not in fieldnames:
			raise RuntimeError("CSV missing 'caption' column")
		
		for row in reader:
			row["caption"] = ""  # Clear caption
			rows.append(row)

	# Write back
	with open(write_path, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

	print(f"Cleared all captions from {write_path}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Fill missing captions using local Ollama Gemma3:4B")
	p.add_argument("--in", dest="in_csv", default="/home/saksham/coding/dl/final_dl.csv", help="Input CSV path")
	p.add_argument("--out", dest="out_csv", default=None, help="Output CSV path (default: overwrite input)")
	p.add_argument("--no-overwrite", dest="overwrite", action="store_false", help="Do not overwrite input (default: overwrite if --out not set)")
	p.add_argument("--overwrite", dest="overwrite", action="store_true", help="Force overwrite input file")
	p.set_defaults(overwrite=True)
	p.add_argument("--limit", type=int, default=None, help="Max unique class_labels to generate (for quick tests)")
	p.add_argument("--clear", action="store_true", help="Clear all captions from the CSV")
	return p.parse_args(argv)


def main() -> None:
	args = parse_args()
	if args.clear:
		clear_captions(in_csv=args.in_csv, out_csv=args.out_csv, overwrite=bool(args.overwrite))
	else:
		fill_captions(in_csv=args.in_csv, out_csv=args.out_csv, overwrite=bool(args.overwrite), limit=args.limit)


if __name__ == "__main__":
	main()
