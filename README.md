# Ontological Differentiation -- Computation Pipeline

Source code, pipeline, and curated corpora for computing Ontological Differentiation (OD) measures across linguistic corpora.

**Author:** P. García-Cuadrillero

This repository contains the complete pipeline for computing Strong Ontological Differentiation (SOD) and Termination Level (TL) across linguistic corpora, together with the scripts used to generate and curate the six corpora used in our research. The pipeline and its outputs serve as the computational foundation for multiple publications based on the OD framework. This repository is under a license Creative Commons Attribution 4.0 International (CC BY 4.0).

## Overview

Ontological Differentiation (OD) formalizes semantic distance through recursive definitional analysis. Given a lexicon where every word is defined in terms of other words in the same set, SOD measures how much unique (uncancelled) content remains when two words' definitions are recursively expanded until convergence. The Termination Level (TL) records the recursive depth at which convergence occurs.

This pipeline computes SOD and TL for all ~N²/2 word pairs in each corpus (N ≈ 16,000–18,000), totaling ~10⁸ pairs per corpus, and performs intra- and inter-corpus statistical analysis.

## Repository Structure

```
.
├── unified_pipeline.sh              # Main orchestrator (Slurm-based, CeSViMa HPC)
├── curated_corpora/                 # The six curated corpora (ready for analysis)
│   ├── curated_ground_filtered.txt
│   ├── curated_random_removal.txt
│   ├── curated_targeted_removal.txt
│   ├── curated_merriam_webster.txt
│   ├── curated_ai_generated.txt
│   ├── curated_null_model.txt
│   └── common_vocabulary.txt
│
├── # --- Data Generation (Pre-Pipeline) ---
├── scrape_wiktionary_full.py        # Scrapes Simple English Wiktionary
├── wiktionary_entries.py            # Processes raw Wiktionary entries
├── wiktionary_definitions.py        # Extracts and cleans definitions
├── wiktionary_validator.py          # Validates extracted entries
├── merriam_webster_polysemic_scraper.py  # Scrapes Merriam-Webster (polysemic)
├── validate_mw_definitions.py       # Validates MW definitions
├── filter_proper_nouns.py           # Filters proper nouns from vocabulary
├── build_valid_vocabulary.py        # Builds the shared valid vocabulary
├── clean_gf_corpus.py               # Cleans the Ground Filtered corpus
├── clean_definitions_with_logging.py # Definition cleaning with logging
├── generate_corpora.py              # Generates GF, Random Removal, Targeted Removal
├── create_null_model.py             # Creates the degree-preserving null model
├── generate_ai_corpus.py            # Generates AI corpus (cluster version)
├── generate_ai_corpus_local.py      # Generates AI corpus (local version)
├── curate_all_corpora.py            # Iterative pruning for self-containedness
│
├── # --- Phase 3: OD Calculations ---
├── run_od_analysis.py               # Core OD engine (SOD pairs + GOD scoring)
│
├── # --- Phase 4: Intra-Corpus Analysis ---
├── step_python_indexer.py           # Builds per-word index from pairs results
├── merge_index_parts.py             # Merges distributed index parts
├── run_fast_agreement.py            # Level coincidence (agreement) analysis
├── aggregate_reports.py             # Aggregates agreement results
├── run_semantic_difference.py       # Semantic difference analysis
├── aggregate_diff_summary.py        # Aggregates semantic diff results
│
├── # --- Phase 5: Inter-Corpus Analysis ---
├── step3_python_eod_worker.py       # EOD (inter-corpus OD) calculation
├── step4_python_diff_worker.py      # Pairwise SOD/TL differences between corpora
├── step5_generate_summary.py        # Final summary report + log histograms
├── step5a_python_loghist_worker.py  # Log-binned histogram worker
│
├── # --- Phase 6: Dynamics Analysis ---
├── step6_dynamics_worker.py         # SOD percentile walks (dynamics)
│
└── # --- Slurm Job Scripts ---
    ├── sbatch_*.slurm               # Individual Slurm job scripts (legacy)
    └── master_pipeline.slurm        # Legacy master submission script
```

## Corpora

The six corpora share a common vocabulary basis (nouns, adjectives, and verbs in lemmatized form) and satisfy the self-containedness constraint: every token in a definition also exists as a headword.

| Corpus | Source | N (words) | Avg. Def. Len. |
|--------|--------|-----------|----------------|
| Ground Filtered | Simple English Wiktionary (first sense) | 17,981 | 4.50 |
| Random Removal | GF with one random token removed per definition | 17,974 | 4.30 |
| Targeted Removal | GF with 20 high-frequency words removed | 17,699 | 3.38 |
| Complex (MW) | Merriam-Webster (up to 2 polysemic senses fused) | 16,834 | 4.83 |
| AI Generated | GPT-5.4 (distribution-controlled generation) | 15,490 | 3.54 |
| Null Model | Degree-preserving configuration model of GF | 17,987 | 4.58 |

### Corpus file format

Each corpus file is a plain-text dictionary with one entry per line:

```
word: token1 token2 token3
```

where `word` is the headword and `token1 token2 token3` is its definition (space-separated nouns, adjectives, and base-form verbs only).

## Pipeline

The pipeline is designed for Slurm-based HPC clusters. The orchestrator `unified_pipeline.sh` manages job submission, monitors completion, aggregates results, and handles resumption.

### Phases

| Phase | Description | Key Script |
|-------|-------------|------------|
| **Phase 3** | SOD + GOD computation for all ~N²/2 pairs per corpus | `run_od_analysis.py` |
| **Phase 4** | Per-word indexing, level coincidence, semantic difference | `step_python_indexer.py`, `run_fast_agreement.py`, `run_semantic_difference.py` |
| **Phase 5** | Inter-corpus EOD, pairwise TL/SOD differences, summary | `step3_python_eod_worker.py`, `step4_python_diff_worker.py` |
| **Phase 6** | Dynamics analysis (SOD percentile walks) | `step6_dynamics_worker.py` |

### Running the pipeline

```bash
# Show current status
./unified_pipeline.sh --status

# Run all phases
./unified_pipeline.sh --run-od --run-analysis --run-eod --run-dynamics

# Run only SOD computation (Phase 3)
./unified_pipeline.sh --run-od

# Customize resources
./unified_pipeline.sh --run-od --jobs 40 --cores 16 --time 48:00:00 --mem 24G
```

### Output structure

Each corpus produces a results directory:
```
results_curated_<corpus_name>/
├── pairs_results.txt         # All pairwise SOD scores and TL values
├── single_god_results.txt    # Per-word GOD scores
└── indexed_pairs_data/       # Per-word indexed pair data (for Phase 4+)
```

The `pairs_results.txt` file contains one line per word pair:
```
idx1 idx2 sod_score termination_level god_score
```

## Pre-Pipeline: Generating the Corpora

If you want to regenerate the corpora from scratch rather than using the provided curated files:

1. **Scrape Wiktionary:**
   ```bash
   python scrape_wiktionary_full.py
   python wiktionary_definitions.py
   ```

2. **Scrape Merriam-Webster:**
   ```bash
   python merriam_webster_polysemic_scraper.py
   ```

3. **Generate derived corpora:**
   ```bash
   python generate_corpora.py        # Creates GF, RR, TR
   python create_null_model.py       # Creates null model
   python generate_ai_corpus_local.py  # Creates AI corpus (requires OpenAI API key)
   ```

4. **Curate all corpora** (iterative pruning for self-containedness):
   ```bash
   python curate_all_corpora.py
   ```

## Requirements

**Cluster (pipeline):**
- Python 3.10+
- NumPy
- Standard library (multiprocessing, itertools, collections)

**Local (data generation):**
- Python 3.10+
- `requests`, `beautifulsoup4` (web scraping)
- `openai`, `tiktoken` (AI corpus generation)
- `tqdm` (progress bars)
- `numpy`, `pandas`, `scipy`, `networkx` (analysis)

## Related Publications

The OD framework was introduced in:

```bibtex
@article{GarciaCuadrillero2026PRE,
  title   = {Ontological Differentiation},
  author  = {Garc{\'\i}a-Cuadrillero, Pablo and Revuelta, Fabio and Capit{\'a}n, Jos{\'e} A.},
  journal = {Physical Review E},
  year    = {2026}
}
```

This pipeline was used to produce the results in:

```bibtex
@article{GarciaCuadrillero2026Signature,
  title   = {The Semantic Structural Signature of a Linguistic Model:
             Ontological Differentiation as a Semantics-Based Analytical Framework},
  author  = {Garc{\'\i}a-Cuadrillero, Pablo and Revuelta, Fabio and Capit{\'a}n, Jos{\'e} A.},
  year    = {2026},
  note    = {Submitted}
}
```

## Acknowledgments

The code in this repository was written by P. García-Cuadrillero with the assistance of Claude Opus 4.6 (Anthropic). All code was reviewed, verified, and directed by the author, who assumes full responsibility for its correctness and design.

Computing resources were provided by the Magerit Supercomputer at the Universidad Politécnica de Madrid (CeSViMa).

## License

This project is licensed under the MIT License.
