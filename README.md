# Leader Algorithms

Research code from my bachelor's thesis on Leader-family clustering algorithms, with Gower similarity for mixed-type data and medoid-based cluster representatives.

The project implements several variants and explores their behaviour through experiments on input order, running time and clustering as a preprocessing step for classification.

## Implementation

`leader_implementation.py` contains the `LeaderAlgorithms` class:

- `Leader` and `Leader2`.
- `Leader_Medoid`, `Leader2_Medoid` and `Leader3_Medoid`.
- Gower and Euclidean similarity functions.
- Feature weights and metadata for binary, categorical, integer and continuous variables.
- Cluster summaries and visualisations, including similarity distributions and simplified silhouette plots.

The implementation uses Python, NumPy, pandas and Numba. Plotting uses Matplotlib, Seaborn and NetworkX.

## Where to start

| File | Purpose |
| --- | --- |
| `study_case.ipynb` | Worked comparison of the algorithm variants using the UCI Heart Disease dataset. |
| `order_dependency.ipynb` | Experiments on sensitivity to observation order. |
| `time_complexity.ipynb` | Running-time experiments with different dataset sizes and feature counts. |
| `clustering_for_classification.ipynb` | Experiment on clustering as a step towards classification. |
| `auxiliary_plots.ipynb` | Additional thesis figures, including plots using the supplied timing CSV files. |
| `leader_implementation.py` | Algorithm implementations and plotting methods. |

## Local setup

From the repository root, create an isolated Python environment and install the dependencies used by the implementation and notebooks:

```bash
python -m venv .venv
# Activate the environment for your operating system before continuing.
python -m pip install numpy pandas numba category-encoders networkx matplotlib seaborn ucimlrepo jupyterlab
python -m jupyter lab
```

Open `study_case.ipynb` for the main example. The UCI dataset is fetched through `ucimlrepo`, so this notebook needs internet access.

The repository does not currently pin dependency versions. The commands above describe the dependencies found in the code, rather than a tested, locked environment.

## API example

```python
import pandas as pd
from leader_implementation import LeaderAlgorithms

# A small synthetic mixed-type dataset, independent of the thesis data.
data = pd.DataFrame({
    "measurement": [1.0, 1.2, 4.8, 5.0],
    "group": ["a", "a", "b", "b"],
})
metadata = pd.DataFrame({
    "name": ["measurement", "group"],
    "type": ["Continuous", "Categorical"],
})

model = LeaderAlgorithms(
    data,
    metadata,
    similarity_func="gower",
    seed=42,
    verbose=0,
)
n_clusters, leaders, labels = model.Leader(s_min=0.8, verbose=0)
```

The feature names and their order in `metadata` must match the columns in `data`. `s_min` is the minimum similarity threshold used by the clustering algorithm. The first call can include Numba compilation time.

## Scope and limitations

This is thesis research code, not a packaged or production-supported library. The experiments study specific datasets and settings; they should not be read as a general claim that one variant always performs better.

There is no automated test suite or locked environment in the current repository. Before reusing the code, check the input assumptions, reproduce the relevant notebook and separate compilation time from execution time when measuring performance.
