# XeniumLGCP

## 1. Installation

### Prerequisites
* **Hardware**: An NVIDIA GPU is required.
* **Python**: Version 3.8 or higher.
* **Google Colab Users**: The environment usually comes with `cupy-cuda12x` pre-installed. You can skip the installation step if it is already present.

### Step-by-Step Setup
1. Clone or download this repository.
2. Install the dependencies using the requirements file:

```bash
pip install -r requirements.txt
```

## 2. Input Data Format

Your data must be a CSV file (e.g., `transcripts.csv`) containing at least these three columns:

- **feature_name**: Gene name (e.g., `"CD3E"`, `"CD19"`)
- **x_location**: X coordinate (microns)
- **y_location**: Y coordinate (microns)

---


## 3. Tutorial: Co-localization Analysis

This example demonstrates how to determine if **Gene A (Target)** co-localizes with **Gene B (Covariate)**.

---

### Step 1: Initialize and Load Data

```python
import XeniumLGCP

# Initialize the model
# mesh_max_edge: Controls resolution (lower = finer mesh).
model = XeniumLGCP.XeniumLGCP(mesh_max_edge=20.0, verbose=True)

# Load your transcripts file
model.load_data("transcripts.csv")
```

### Step 2: Fit the Co-localization Model

We fit the model treating the **Covariate Gene** as a continuous spatial field that influences the intensity of the **Target Gene**.

```python
# target_gene: The points we are modeling (e.g., T-cells)
# covariate_gene: The spatial predictor (e.g., B-cells)
model.fit(target_gene="CD3E", covariate_gene="CD19")
```

### Step 3: Interpret Results

Check the estimated coefficients to quantify the relationship.

```python
# Retrieve coefficients from the GPU
beta = model.model_.beta_.get()

print(f"Intercept (Baseline Density): {beta[0]:.3f}")
print(f"Co-localization Effect:       {beta[1]:.3f}")
```

### Step 4: Visualizations
Generate plots to see the fitted intensity, the underlying latent field (spatial residual), and the covariate density.

```python
# 1. Visualize maps
model.plot_results()

# 2. Check convergence (optional)
model.plot_loss()
```