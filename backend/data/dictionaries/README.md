# Dataset Dictionary Pack

This folder defines the **canonical complaint dictionary format** for the grievance urgency system.

## Files

- `complaint_record.schema.json`  
  JSON Schema (Draft 2020-12) for validating a full complaint record.

- `complaint_record.template.json`  
  Minimal template showing required and optional fields with placeholders.

- `complaint_record.example.json`  
  Realistic filled example aligned with pipeline stages.

- `dataset_columns_dictionary.md`  
  Data dictionary for dataset creators (column name, type, required, meaning).

## Intended Usage

1. **Data ingestion**: map raw source columns into canonical fields.
2. **Validation**: enforce schema using `complaint_record.schema.json`.
3. **Persistence**: store records as JSON/JSONL using this structure.
4. **Model pipeline**: each stage writes only to its corresponding object (`preprocessing`, `model_prediction`, `semantic_analysis`, etc.).

## Validation Example

```bash
python -m json.tool data/dictionaries/complaint_record.example.json > /dev/null
python -m json.tool data/dictionaries/complaint_record.template.json > /dev/null
python -m json.tool data/dictionaries/complaint_record.schema.json > /dev/null
```
