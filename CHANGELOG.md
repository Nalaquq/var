# Changelog

All notable changes to the MVFoul project are documented here.

---

## 2026-04-09

- Created `notebooks/00_unified_pipeline.ipynb` — single notebook combining dataset statistics (NB1) and three-baseline training (NB2) into one linear flow
- Fixed data handoff bug: NB2 never loaded CSVs from NB1 (different column schemas, no `pd.read_csv` calls); unified notebook eliminates the file transfer entirely
- Fixed `train_df` variable name collision between statistics section and training section (renamed to `stats_train_df` in Part 3)
- Rewrote `docs/07-google-colab.md` — beginner-friendly, oriented around the unified notebook, added troubleshooting table
- Updated `README.md` — recommended workflow now points to the unified notebook; original NB1/NB2 kept for reference
- Original notebooks (`01_dataset_statistics.ipynb`, `02_three_baselines.ipynb`) preserved unchanged

---

## 2026-04-08

- Completed Phase 0: ran `01_dataset_statistics.ipynb` in Google Colab
- Downloaded CSV outputs to `notebooks/outputs/`
- Added `docs/07-google-colab.md` — guide for uploading/downloading files in Colab
- Updated README to reference new Colab guide and `notebooks/outputs/` directory
- Added CHANGELOG.md to track project progress
