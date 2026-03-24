# Chapter 6 - Dimension Reduction and Geographic Segmentation

> High-dimensional demographic data made visible. You will learn to compress dozens of ACS variables into interpretable maps of geographic and demographic similarity.

> Full runnable code for all examples is in `examples/chapter-06/`.

```{admonition} Who is this for?
If you finished Chapters 1 and 2 (regression/classification and cross-validation), you are ready. No prior knowledge of ACS variable structures is required.
```

## Learning goals

- Explain the curse of dimensionality and why it matters for survey stratification.
- Apply PCA to county-level ACS data and read loadings, scores, and explained variance.
- Use t-SNE to produce nonlinear 2D maps of demographic similarity.
- Use UMAP for fast, structure-preserving embeddings.
- Interpret demographic clusters in geographic and policy terms.
- Connect dimension reduction to stratification design and nonresponse analysis.
- Communicate segmentation findings to non-technical leadership.

```{admonition} Why this matters for federal statistics
:class: tip
Federal surveys cover thousands of geographies with dozens of correlated demographic variables. Dimension reduction helps you:
- Identify the primary axes of demographic variation across counties or tracts.
- Design stratification schemes that group similar areas to reduce variance.
- Detect geographic coverage gaps in nonresponse patterns.
- Produce briefing visualizations that non-technical audiences can interpret intuitively.
- Build hot-deck imputation classes by grouping similar records.
```

---

## 1. The data: synthetic county ACS profiles

The examples in this chapter use a synthetic county-level dataset that mimics ACS 5-year estimates: 400 counties, 15 demographic and economic variables per county. Three latent demographic profiles generate the data — urban, suburban, and rural — but those labels are withheld during analysis and only used afterward to interpret results.

This mirrors the real workflow. You run unsupervised methods without knowing the "right answer," then examine the outputs to see whether the discovered structure aligns with what domain experts already believe about geographic variation.

The full dataset generation is in `examples/chapter-06/01_synthetic_county_data.py`. The 15 variables are:

```{code-block} python
FEATURE_COLS = [
    "median_age", "pct_bachelors", "median_hh_income", "pct_poverty",
    "pct_owner_occupied", "pct_employed", "pct_under18", "pct_over65",
    "pct_hispanic", "pct_foreign_born", "pct_renter", "pop_density_log",
    "median_gross_rent", "pct_no_vehicle", "pct_broadband",
]
```

Run `01_synthetic_county_data.py` before any other example. It writes `county_data.csv` to the `examples/chapter-06/` directory.

---

## 2. The curse of dimensionality

With 15 ACS variables we can still enumerate pairwise plots. In practice, ACS 5-year estimates include hundreds of variables across age, sex, race, income, housing, employment, and commute. Adding more dimensions makes the data sparser — it becomes harder to define "similar counties" and harder to sample densely enough to generalize.

The specific problem is *distance concentration*. In high-dimensional space, pairwise Euclidean distances between points cluster together: the nearest and farthest neighbors become nearly indistinguishable. The figure below shows this with uniform random points at 2, 15, and 100 dimensions. At 2 dimensions, distances vary widely. At 100 dimensions, the distribution narrows sharply — everything is roughly the same distance from everything else.

This matters because clustering algorithms (K-means, hot-deck nearest-neighbor matching) depend on meaningful distances. When distances concentrate, clusters lose their separation. Dimension reduction solves this by finding a low-dimensional representation that preserves the structure we care about.

`examples/chapter-06/02_curse_of_dimensionality.py` produces the distance concentration histograms and a pair plot for a 6-variable subset. With 15 variables there are 105 pairs — impractical to scan. With 100 variables: 4,950 pairs.

As dimensionality grows:
- The mean distance increases while the variance shrinks relative to the mean.
- "Nearest neighbors" become nearly as far as "farthest neighbors."
- Clustering algorithms rely on meaningful distances — high-dimensional spaces break them.

---

## 3. Supervised vs. unsupervised

Chapters 1–4 covered supervised learning: we had a target variable (income, response status) and trained models to predict it. This chapter switches to *unsupervised learning*.

- *Dimension reduction*: compress $p$ features to $k \ll p$ coordinates without using any label. Goal: preserve structure for visualization or downstream analysis.
- *Clustering*: group observations based on distance or density, without labeled groups. Goal: discover natural segments in the data.

We have `profile` labels in our synthetic data, but we only use them to color plots and interpret results — not during fitting. This mirrors the real workflow: you run unsupervised methods, then label the clusters after the fact by examining what is in them.

```{admonition} Bounded agency
:class: warning
Unsupervised methods find mathematical patterns. Whether a cluster is meaningful for policy purposes requires domain expertise. A cluster of counties that are statistically similar may span very different administrative boundaries, histories, or needs. Analysts present clusters; humans decide how to use them.
```

---

## 4. Principal Component Analysis (PCA)

PCA is a linear method that finds a new coordinate system where:
- The axes (principal components) are orthogonal.
- PC1 points in the direction of greatest variance, PC2 in the direction of second greatest variance, and so on.
- Each component is a weighted combination (linear combination) of the original features.

**Math:**

1. Standardize the feature matrix $X$ so each column has mean 0 and variance 1: call this $\tilde{X}$.
2. Compute the covariance matrix $S = \frac{1}{n-1}\tilde{X}^\top \tilde{X}$.
3. Eigen-decompose $S$ (or equivalently, SVD of $\tilde{X}$): the eigenvectors are the principal directions (loadings), the eigenvalues are the variances explained.
4. Project the data: $Z = \tilde{X} V_k$ gives the score matrix — where each county sits in the new $k$-dimensional space.

What this means for county data:
- Each county is described by 15 ACS variables.
- PCA finds new axes that are linear combinations of those 15 variables.
- PC1 might be something like "economic disadvantage" (high poverty, low income, low education loading together).
- PC2 might capture "urbanicity" (population density, renter fraction, no-vehicle rate).
- We can then plot 400 counties in 2D using only PC1 and PC2, capturing most of the information.

### 4.1 Scree plot and explained variance

The scree plot (bar chart of variance explained per component) and the cumulative explained variance plot are the first things to examine after fitting PCA.

The scree plot for our 400-county dataset shows a sharp drop after PC1 and a more gradual decline afterward — an "elbow" that suggests the first 3–4 components capture the dominant structure. The cumulative plot shows that 3 components explain about 68% of variance, and 5 components reach 80%.

*What to look for:*
- Where is the elbow? That is the natural compression point.
- Does the first component explain a dominant share (> 30%), or is variance spread evenly across many components? Evenly spread variance suggests no single strong demographic axis — worth investigating before proceeding.
- How many components do you need for 80% variance? That is a common threshold for downstream analysis (stratification, imputation class construction).

`examples/chapter-06/03_pca.py` generates the scree plot and cumulative variance figure.

```{code-block} python
pca_full = PCA(random_state=42)
pca_full.fit(X_std)
explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)

n80 = np.searchsorted(cumulative, 0.80) + 1
print(f"Components for 80% variance: {n80}")
```

### 4.2 Loadings: what does each component measure?

A *loading* is the weight of each original variable in a principal component. Large positive loadings mean "high values of this variable push counties toward positive PC1." Large negative loadings push toward negative PC1.

```{admonition} Interpreting loadings
:class: tip
Read a cluster of same-signed loadings together — they tend to represent a latent concept like "economic disadvantage" or "urbanicity." If PC1 mixes unrelated variables with no clear thematic pattern, the data may lack clean structure and PCA may not compress well.
```

For the county dataset, PC1 typically captures an economic gradient: income, education, and poverty load on the same axis with opposite signs (high income and high education push toward positive PC1; high poverty pushes negative). PC2 captures urbanicity: population density, renter fraction, and no-vehicle rate together. These are interpretable patterns that match what demographers already know about county variation.

`examples/chapter-06/03_pca.py` produces horizontal bar charts of PC1 and PC2 loadings.

```{code-block} python
loadings = pd.DataFrame(
    pca_2.components_.T,
    index=FEATURE_COLS,
    columns=["PC1", "PC2"],
)
```

### 4.3 PCA score plot: counties in 2D

The score plot projects all 400 counties onto the PC1–PC2 plane. When colored by the known demographic profile, the three groups (urban, suburban, rural) separate clearly along the PC axes — confirming that PCA recovered the underlying structure without being told what the groups were.

Coloring the same plot by median household income rather than profile reveals that the PC1 axis tracks income almost exactly. This is the interpretive payoff: the axis label may say "PC1," but we now know it means "economic position."

`examples/chapter-06/03_pca.py` produces both panels.

### 4.4 PCA biplot: scores and loadings together

The biplot overlays the county positions (scores) with arrows for each ACS variable (loadings). It is the standard visualization demographers use to characterize geographic segments in briefings.

Reading a biplot:
- Points close together are demographically similar.
- Variables pointing in similar directions are positively correlated.
- A variable arrow pointing toward a cluster of counties means those counties score high on that variable.
- Variable arrows pointing in opposite directions are negatively correlated (e.g., `pct_poverty` and `median_hh_income` point in opposite directions along PC1).

`examples/chapter-06/03_pca.py` generates the biplot.

```{admonition} Reading a biplot
:class: note
Points close together are demographically similar. Variables pointing in similar directions are positively correlated. A variable pointing toward a cluster of counties means those counties score high on that variable. This is the standard visualization that demographers use to characterize geographic segments in briefings.
```

### 4.5 Reading a PCA output: what to look for

When you receive a PCA output — or are evaluating one someone else produced — ask these four questions before drawing any conclusions.

*1. How much variance do the first 2–3 PCs explain?*

Less than 50% total for 3 components is a warning sign. It does not mean PCA failed, but it does mean the 2D visualization is a rough approximation. Interpret with caution and note the limitation when briefing leadership. The scree plot should show this clearly.

*2. Are the loadings interpretable?*

If PC1 mixes unrelated variables — say, `median_age`, `pct_renter`, and `pct_broadband` all loading similarly with no clear theme — the data may lack clean structure or may need different preprocessing (log transforms, outlier removal). Well-structured ACS data usually produces interpretable first components.

*3. Do the score plots separate meaningful groups?*

Color the score plot by known attributes: region, urbanicity, state, or any variable you trust. If meaningful geographic groups land in distinct PCA regions, the components are capturing real variation. If all colors are scrambled uniformly, the components may not be useful for the intended application.

*4. Are there outliers in the score plot?*

Counties far from the main cluster in PCA space deserve investigation before clustering. K-means centroids are pulled toward outliers, which can distort the entire cluster solution. Investigate: are these data quality issues (extreme values due to small county population), genuine outliers (unusual economic conditions), or data entry errors?

### 4.6 PCA for stratification design

PCA components can serve directly as stratification variables. The first few PCs capture the dimensions along which counties vary most — and stratification aims to create homogeneous groups to reduce variance. Counties in the same PCA cluster have similar demographic profiles, so they are natural stratum candidates.

Cutting PC1 and PC2 into terciles produces a 3×3 stratification grid (9 strata). For the county dataset, this reduces within-stratum variance on median household income by roughly 35–45% compared to no stratification — a meaningful gain achievable without any domain-specific variable engineering.

```{code-block} python
pc1_terciles = pd.qcut(scores_2d[:, 0], q=3,
                       labels=["PC1_low", "PC1_mid", "PC1_high"])
pc2_terciles = pd.qcut(scores_2d[:, 1], q=3,
                       labels=["PC2_low", "PC2_mid", "PC2_high"])
df["pca_stratum"] = pc1_terciles.astype(str) + "_" + pc2_terciles.astype(str)

total_var = df["median_hh_income"].var()
within_var = df.groupby("pca_stratum")["median_hh_income"].var().mean()
reduction = (1 - within_var / total_var) * 100
```

`examples/chapter-06/08_operations.py` contains the full stratification variance reduction analysis.

---

## 5. Nonlinear embeddings: t-SNE and UMAP

PCA is linear — it finds straight-line combinations of features. But demographic data often lives on curved surfaces in high-dimensional space. Two counties might be similar in many ways but very different in one that PCA would miss. Nonlinear methods like t-SNE and UMAP find 2D representations that can follow these curved structures.

### 5.1 t-SNE

*Idea:* Define probabilities that two counties are neighbors in the original 15-dimensional space (using Gaussian kernels). Then find a 2D layout whose neighbor probabilities match. Points that are close in 15D will be close in 2D.

*Math:*
- Convert distances to conditional probabilities with a Gaussian kernel per point:
  $p_{j\mid i}=\frac{\exp(-\|x_i-x_j\|^2 / 2\sigma_i^2)}{\sum_{k\ne i}\exp(-\|x_i-x_k\|^2 / 2\sigma_i^2)}$
- Symmetrize: $P_{ij} = (p_{j\mid i} + p_{i\mid j}) / 2n$
- In 2D, use a Student-t kernel for $Q_{ij}$.
- Minimize KL divergence $\text{KL}(P \| Q)$ by gradient descent.

*Key hyperparameter:* `perplexity` — roughly, the effective number of neighbors (5 to 50 for most datasets). Start at 30 and test sensitivity.

*Cautions:* t-SNE axes are not interpretable. The scale and orientation mean nothing. Only the relative positions of clusters matter. t-SNE cannot embed new data points without refitting from scratch.

`examples/chapter-06/04_tsne.py` produces the main embedding and a perplexity sensitivity comparison at values 5, 30, and 50. If the cluster structure looks qualitatively similar across all three perplexity values, the structure is robust. If it changes substantially, interpret cautiously.

### 5.2 UMAP

UMAP (Uniform Manifold Approximation and Projection) builds a weighted k-nearest-neighbor graph in the original space and finds a 2D layout that preserves that graph's structure. Compared to t-SNE:
- Faster on large datasets.
- Better preservation of global structure (relative distances between clusters are more meaningful).
- Supports `.transform()` — you can fit on training counties and embed new ones without refitting.
- Supports custom distance metrics (useful for binary features).

*Key hyperparameters:*
- `n_neighbors`: local vs. global balance. Small = focus on fine structure. Large = preserve global shape. Start at 15.
- `min_dist`: cluster tightness. Small = tight blobs. Large = spread-out layout.

`examples/chapter-06/05_umap.py` produces the main embedding and an `n_neighbors` sensitivity comparison.

### 5.3 PCA vs. t-SNE vs. UMAP: when to use which

The table and decision framework below match each method to its strongest federal statistics use case.

| Method | Axes interpretable | Distances meaningful | Captures global structure | Best for |
|---|---|---|---|---|
| PCA | Yes | Yes | Yes | Stratification, variance summarization |
| t-SNE | No | No | No | Cluster visualization for briefings |
| UMAP | No | Approx | Partial | Exploratory cluster visualization |
| K-means | — | Yes (feature space) | Yes | Operational segmentation, imputation classes |

*Decision framework for federal applications:*

- *Stratification design* → PCA. Interpretable axes mean loadings tell you what each stratum represents. PC1 = "economic position" is actionable; "UMAP-1" is not.
- *Executive briefing visualization* → t-SNE or UMAP. Cleaner cluster separation makes it easier for non-technical audiences to see that geographic segments exist. Label the clusters by name, not by axis values.
- *Hot-deck imputation classes* → K-means on original features (not on any embedding). Distances in embedding space are distorted and do not preserve the feature-space similarity that makes hot-deck imputation valid.
- *Exploratory analysis before modeling* → try PCA first. If structure is clear from the scree plot and score plot, stop. If not, try t-SNE or UMAP to see whether clusters exist at all before committing to a segmentation.

`examples/chapter-06/06_method_comparison.py` produces a side-by-side panel of all three methods on the same 400-county dataset. PCA clusters overlap more than t-SNE/UMAP because PCA optimizes for global variance, not local separation — but its axes tell you *why* counties differ, which the other methods cannot.

---

## 6. Clustering: K-means and hierarchical

Dimension reduction shows us there is structure. Clustering formalizes that structure by assigning each county to a group.

### 6.1 K-means clustering

K-means partitions counties into $k$ groups by minimizing the within-cluster sum of squared distances to the cluster centroid. It requires specifying $k$ in advance.

```{admonition} Important: cluster on features, visualize on embeddings
:class: warning
Always run K-means on the original standardized features, not on the 2D t-SNE/UMAP coordinates. The 2D embedding distorts distances. Use PCA, t-SNE, or UMAP only to *visualize* the clusters after they are assigned. This is one of the most common errors in unsupervised analysis pipelines.
```

Choosing $k$: two heuristics guide the choice.

The *elbow method* plots inertia (within-cluster sum of squared errors) vs. $k$. Look for the bend where adding another cluster gives diminishing returns. The elbow is often ambiguous — treat it as a starting point.

The *silhouette score* measures how similar each point is to its own cluster compared to neighboring clusters. Values range from −1 to +1; higher is better. The $k$ with the highest silhouette score is a reasonable default.

Neither heuristic is ground truth. Domain expertise should govern the final choice. A $k=4$ solution that maps cleanly onto operationally meaningful geographic types is more useful than a $k=7$ solution that maximizes a mathematical criterion but produces clusters no one can name.

```{code-block} python
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_std)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_std, labels))

best_k = K_range[np.argmax(sil_scores)]
```

`examples/chapter-06/07_clustering.py` produces the elbow and silhouette plots.

### 6.2 Cluster profiles: what do the clusters mean?

Assigning cluster labels is a mathematical operation. *Naming* the clusters — and deciding whether they are operationally meaningful — is a human judgment.

The cluster profiling heatmap (generated by `07_clustering.py`) shows normalized variable means for each cluster. Each row is a cluster; each column is an ACS variable. Cells closer to 1 (dark green) indicate a cluster with high values on that variable; cells near 0 (dark red) indicate low values.

Reading example: a cluster with high `median_hh_income`, high `pct_bachelors`, high `pop_density_log`, high `pct_renter`, and low `pct_poverty` is clearly urban. A cluster with high `median_age`, high `pct_over65`, high `pct_owner_occupied`, and low `pop_density_log` is clearly rural-older. Naming these clusters is trivial once the heatmap is in front of you. The algorithm grouped them; you named them.

### 6.3 Hierarchical clustering

Hierarchical clustering does not require specifying $k$ in advance. It builds a tree (dendrogram) of merges. You cut the tree at a height that gives the number of clusters you want, or let the dendrogram guide that decision.

The dendrogram produced by Ward linkage is read top-down: the height of each join indicates how dissimilar the merged groups are. A natural cut point appears where the next merge would combine groups that are substantially more different than the previous merges — a visible gap in the tree.

`examples/chapter-06/07_clustering.py` produces a dendrogram on a 60-county subsample (full dendrogram on 400 counties is too dense to read) and the Adjusted Rand Index (ARI) comparing K-means and hierarchical assignments. High ARI (> 0.8) means both methods agree — the cluster structure is robust to the choice of algorithm.

---

## 7. Connecting clusters to survey operations

### 7.1 Nonresponse analysis

Clusters can reveal which geographic segments have lower response rates. If counties in the urban cluster consistently show lower response, adaptive design resources can be targeted there before data collection begins.

The operational workflow is:
1. Cluster counties on demographic features before the survey field period.
2. Attach historical response rates (from the previous cycle) to each cluster.
3. Flag clusters below a response rate threshold for enhanced outreach or priority follow-up.
4. After the current cycle, validate: did targeted clusters respond at higher rates?

`examples/chapter-06/08_operations.py` simulates this analysis, assigning lower response rates to the urban cluster and showing the detection in a bar chart.

### 7.2 Hot-deck imputation classes

Hot-deck imputation replaces a missing value with an observed value from a "donor" — a similar record. Clusters define the donor pool: only counties in the same cluster are eligible donors, ensuring demographic similarity within the imputation class.

This is the standard approach used in ACS and CPS imputation. The key requirement is that donors share the cluster assignment in the *original feature space*, not in any 2D embedding. Embedding-space distances do not preserve the demographic similarity relationships that justify within-class donation.

`examples/chapter-06/08_operations.py` demonstrates this by simulating 10% missing income values, imputing from within-cluster donors, and comparing the imputed distribution to the true values.

### 7.3 The bounded agency principle in segmentation

Algorithms find mathematical clusters. Naming them and deciding how to use them requires domain expertise. A cluster of "high-poverty rural counties with low internet access and high elderly population" has policy implications that no algorithm can evaluate. The statistician names the cluster; the domain expert decides what to do with it. The algorithm only groups.

This is bounded agency applied to unsupervised learning: the algorithm assists, the human decides what the groups mean and whether to act on them. The risk of ignoring this boundary runs in both directions: acting on clusters that do not make operational sense wastes resources; dismissing clusters that do make sense because an algorithm found them misses real patterns.

Before presenting a cluster solution to leadership or using it in an operational decision, ask yourself: can I describe each cluster in plain language? Would a subject matter expert recognize these groups as meaningful? If the answer to either question is no, do more interpretive work before moving forward.

---

## 8. Evaluating a segmentation proposal

When someone presents a cluster analysis — including your own — these five questions should guide the review.

*1. How was k chosen?*

Elbow plot? Silhouette score? Domain knowledge? "We tried a few and liked 4" is not sufficient. The choice of k is a modeling decision that should be documented and defensible. If the elbow and silhouette disagree, explain which won and why.

*2. Were clusters computed on raw features or on a 2D embedding?*

This is a quick disqualifier. Clustering on t-SNE or UMAP coordinates is incorrect. The 2D embedding distorts distances to achieve visual separation; cluster assignments based on those distorted distances do not correspond to meaningful groups in the original feature space.

*3. Do the cluster profiles make operational sense?*

Can you name each cluster based on its characteristics? If Cluster 2 is "high income, high education, high density, low poverty," that is Urban. If Cluster 2 has no coherent character — it is medium on everything — the cluster may be absorbing residual variance rather than capturing a real group.

*4. Are the clusters stable?*

Run K-means with five different random seeds. If the cluster compositions change substantially (large change in ARI across seeds), the clusters are not stable. Possible causes: $k$ is too large, the data contains no true cluster structure, or outliers are destabilizing centroids. Investigate before using the solution operationally.

*5. Are operationally important subgroups split across clusters?*

If counties in the same state, the same urbanicity tier, or the same historical response-rate category are split across clusters, the solution may create administrative problems. Statistical optimality and operational usability are not always the same thing.

---

## 9. In-class activity

You are advising a regional Census office about adaptive survey design for the next ACS collection cycle. Your task is to evaluate — not just run — a segmentation of 50 simulated counties.

*Provided information (pre-computed):*

The scree plot for the 50-county dataset shows the first 3 PCs explain 68% of variance. The elbow occurs at PC3. A 4-cluster K-means solution has been computed on the original standardized features.

*Cluster profile table (pre-computed):*

| Cluster | Median income | Pct poverty | Pop density (log) | Pct broadband | Pct over 65 |
|---------|--------------|-------------|-------------------|---------------|-------------|
| 0 | ~$72,000 | ~8% | ~7.1 | ~87% | ~12% |
| 1 | ~$44,000 | ~20% | ~4.5 | ~62% | ~22% |
| 2 | ~$58,000 | ~13% | ~5.5 | ~76% | ~17% |
| 3 | ~$38,000 | ~24% | ~3.2 | ~58% | ~26% |

*Questions:*

1. The first 3 PCs explain 68% of variance. The elbow occurs at PC3. How many PCs would you use for stratification? Justify your choice.

2. Name each cluster in the profile table. Which cluster should get priority follow-up resources, and why?

3. A colleague proposes clustering using the t-SNE 2D coordinates instead of the original 8 variables. What is wrong with this approach?

4. The cluster solution changes substantially when you change the random seed. What does this tell you? What would you do?

5. *(Optional)* Run the full pipeline using `examples/chapter-06/09_exercise.py` and verify your cluster naming against the computed profiles.

---

## Key takeaways for survey methodology

- *Dimension reduction reveals structure* that is invisible in pairwise plots of 15+ ACS variables. PCA, t-SNE, and UMAP each compress demographic complexity into interpretable 2D maps.
- *PCA axes are interpretable* through loadings. The first PC often represents an economic gradient (income, education, poverty all loading together). The second often captures urbanicity.
- *t-SNE and UMAP* show cluster separation more clearly than PCA for nonlinear structure, but their axes carry no direct meaning.
- *Always cluster on features, not on 2D embeddings.* The 2D projection distorts distances. Use embeddings only for visualization.
- *Clusters connect to operations.* The same segmentation that explains demographic variation also:
  - Guides stratification (group similar geographies into strata).
  - Identifies nonresponse hotspots (which clusters have systematically low response rates).
  - Defines hot-deck imputation classes (within-cluster donors are demographically similar).
- *Silhouette score and the elbow method* are heuristics, not ground truth. Domain expertise should inform the final choice of k.
- *The bounded agency principle:* algorithms cluster; humans name, validate, and decide. A segmentation that a statistician cannot explain to leadership should not be used operationally.

```{admonition} How to explain these methods to leadership
:class: tip
**What problem are we solving?** We have 400 counties described by 15 demographic variables. No one can examine 6,000 numbers and spot patterns. We need a way to see the big picture.

**What does PCA do?** It finds the two or three main axes of variation — the dimensions along which counties differ the most — and uses those to draw a map. Think of it as creating the "most important" demographic summary scores.

**What do the clusters mean?** Counties in the same cluster are demographically similar. The map shows three broad groups in our data: urban high-education counties, suburban moderate-income counties, and rural older-population counties.

**Why does this matter for the survey?** First, we can design strata based on these groups — sampling proportionally within each reduces estimation error. Second, the map shows which cluster has the lowest response rate, so we know where to focus adaptive design follow-up before we even start collecting data.

**Is this automatic?** The algorithm finds the mathematical clusters. We still have to name them, validate that they make geographic sense, and decide how to use them operationally. The model suggests; analysts decide.
```
