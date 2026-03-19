# Full-Field Data-Driven Material Parameter Estimation in Linear Elasticity

---

## 📋 Overview

This project explores the identification of material parameters for **linear elastic materials** using full-field displacement data through two complementary approaches:

1. **FEM-Based Parameter Identification** — using [DOLFINx](https://github.com/FEniCS/dolfinx) as the finite element solver
2. **Gaussian Process (GP) Surrogate Model** — to accelerate the parameter identification process

The primary goal is to calibrate the **shear modulus (G)** and **bulk modulus (K)** by minimizing the difference between numerical and experimental (DIC) displacement data.

---

## 🧪 Problem Setup

- **Specimen:** Dog bone specimen with a central hole
- **Boundary Conditions:**
  - Dirichlet BC at `x = 0.0 m` → fixed displacement `ū = 0`
  - Neumann BC at `x = 0.1 m` → traction `T̄ = [106.26, 0]ᵀ N/mm²`
- **Mesh:** Triangular elements, size `0.5×10⁻² m`, 148 nodes, 304 elements
- **Experimental Data:** DIC measurements from a tensile test on an S235 steel specimen

---

## ⚙️ Methodology

### 1. Forward Problem (FEM)
The static linear elasticity problem is solved using the **weak form** of the balance of linear momentum in DOLFINx:

$$
\int_{\Omega} \mathbf{P} : \nabla \delta \mathbf{u} \, dx = \int_{\Gamma_N} \bar{\mathbf{T}} \cdot \delta \mathbf{u} \, ds
$$

with the constitutive relation:

$$
\mathbf{P} = K \, \text{tr}(\boldsymbol{\varepsilon}(\mathbf{u})) \mathbf{I} + 2G \left(\boldsymbol{\varepsilon}(\mathbf{u}) - \frac{1}{3}\text{tr}(\boldsymbol{\varepsilon}(\mathbf{u})) \mathbf{I}\right)
$$

### 2. Parameter Identification
Parameters are identified by solving the reduced optimization problem:

$$
\boldsymbol{\kappa}^* = \arg\min_{\boldsymbol{\kappa}} \mathcal{L}(\boldsymbol{\kappa})
$$

using a **weighted MSE loss** to account for different displacement scales in x and y directions.

### 3. GP Surrogate Model
A **multi-task Gaussian Process** with an anisotropic RBF kernel (ARD) is trained on 100 FEM simulations to learn the mapping:

$$
\mathcal{F}: (\boldsymbol{\kappa}, \mathbf{x}) \mapsto \mathbf{u}
$$

Once trained, the surrogate replaces expensive FEM calls during optimization.

---

## 📊 Results

### Experimental Displacement Field
> *Raw DIC displacement data used for calibration*

<!-- Add experimental displacement image here -->
![Experimental Displacement](figures/images/experimental_displacement.pdf)

---

### FEM-Based Identification

| Parameter | Reference Value | FEM Solution | Relative Error |
|-----------|----------------|--------------|----------------|
| G [GPa]   | 71.10          | 68.90        | 3.00 %         |
| K [GPa]   | 109.10         | 83.10        | 24.01 %        |

> *FEM converged in **128 iterations** (~1.40 min total)*

**FEM Displacement Field:**
<!-- Add FEM displacement field image here -->
![FEM Displacement Field](figures/images/displacement_field.pdf)

**FEM Error Field:**
<!-- Add FEM error field image here -->
![FEM Error](figures/images/error.pdf)

---

### GP Surrogate-Based Identification

| Parameter | Reference Value | FEM Solution | Surrogate Model | GP Error |
|-----------|----------------|--------------|-----------------|----------|
| G [GPa]   | 71.10          | 68.90        | 63.50           | 10.6 %   |
| K [GPa]   | 109.10         | 83.10        | 89.00           | 18.4 %   |

> *GP optimization converged in **76 iterations** (~1.01 sec optimization time)*

**GP Training History:**
<!-- Add training history image here -->
![Training History](figures/images/training_history.pdf)

**GP Predicted Displacement Field:**
<!-- Add GP displacement field image here -->
![GP Displacement Field](figures/images/gp_predictions_displacement_field.pdf)

**GP Error Field:**
<!-- Add GP error field image here -->
![GP Error](figures/images/surrogate_error.pdf)

---

### ⏱️ Computational Performance Comparison

| Metric                          | FEM Approach | GP Surrogate |
|---------------------------------|-------------|--------------|
| Single evaluation time          | 2 s         | 2.5 s        |
| Optimization iterations         | 128         | 76           |
| Optimization time               | 1.40 min    | 1.01 min     |
| Training time                   | N/A         | 30 min       |
| Total time (first run)          | 1.42 min    | ~33 min      |
| Total time (subsequent runs)    | 1.40 min    | 1.01 min     |

---

## 🏁 Conclusion

- The GP surrogate model effectively captures the parameter-to-displacement mapping
- Identified parameters: **G = 6.35×10¹⁰ Pa**, **K = 8.9×10¹⁰ Pa**
- Corresponding material properties: **E ≈ 1.56×10¹¹ Pa**, **ν ≈ 0.23**
- The surrogate approach becomes highly advantageous for **repeated identifications**, reducing optimization time significantly

---

## 📦 Dependencies

- `dolfinx` — FEM solver
- `scikit-learn` — Gaussian Process regression
- `scipy` — Nelder-Mead optimization
- `numpy`, `matplotlib`

---

## 📚 References

- Pierron & Grédiac (2012) — *The Virtual Fields Method*
- Rasmussen & Williams (2006) — *Gaussian Processes for Machine Learning*
- Baratta et al. (2023) — *DOLFINx: The next generation FEniCS problem solving environment*
- Anton et al. (2024) — *Deterministic material parameter identification*

---

## 📄 License

This project is for academic purposes at **Technische Universität Braunschweig**.