# Chapter 6 Code Examples: Dimension Reduction and Geographic Segmentation

Run in order:

1. `01_synthetic_county_data.py` — generate county-level ACS-style data
2. `02_curse_of_dimensionality.py` — distance concentration demonstration
3. `03_pca.py` — PCA: scree plot, loadings, score plot, biplot
4. `04_tsne.py` — t-SNE at multiple perplexity values
5. `05_umap.py` — UMAP embedding
6. `06_method_comparison.py` — side-by-side method comparison
7. `07_clustering.py` — K-means and hierarchical clustering
8. `08_operations.py` — operational applications (stratification, nonresponse, hot-deck)
9. `09_exercise.py` — 50-county exercise and solution

## Requirements

Python 3.9+, numpy, pandas, matplotlib, scikit-learn, scipy

Optional: umap-learn (for UMAP; chapter works without it)

```
pip install numpy pandas matplotlib scikit-learn scipy
pip install umap-learn   # optional
```

## Output files

Each script saves PNG figures to this directory.  Run `01_synthetic_county_data.py`
first — it writes `county_data.csv` which all subsequent scripts read.
