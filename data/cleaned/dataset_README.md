# Cleaned EXACT Dataset Package

Created from:
- Logic_Based_Educational_Queries(2).json
- Physics_Problems_Text_Only(3).csv

## Main files

### Logic
- `logic_cleaned_records.json`: original record structure with `answers-clean`, `premises-FOL-clean`, and quality metadata.
- `logic_flattened_cleaned_all.jsonl`: one question per line, includes raw and cleaned answer.
- `logic_train_safe.jsonl`: safer logic samples for SFT/training.
- `logic_suspect_review.jsonl`: samples that require manual review or should not be used for answer training.

### Physics
- `physics_cleaned_all.csv/jsonl`: all physics rows with `question_clean`, `answer_clean`, `unit_clean`, quality flags.
- `physics_train_safe.csv/jsonl`: safer physics rows for SFT/training.
- `physics_suspect_review.csv/jsonl`: QA/missing/ambiguous/meta-text/multi-target/noisy rows.

### Mixed SFT
- `sft_train_mixed_solver_clean.jsonl`: ready-to-use instruction-tuning style dataset with prompt/completion pairs.

## Summary
- Logic: 411 records -> 808 flattened questions; 644 safe samples; 164 suspect samples; 203 answer labels corrected from explicit explanation cues.
- Physics: 1755 rows; 1329 safe samples; 426 suspect samples.
- Mixed SFT samples: 1961.

## Important caution
This is an automatic, deterministic cleaning pass. High-confidence corrections were applied from explicit explanation patterns and formula checks. Suspect files should still be reviewed before being used as final ground truth.


Patch note: TD003 was detected as a complex disconnected-dielectric capacitor case and kept with its raw answer instead of simple E=0.5CU² correction.
