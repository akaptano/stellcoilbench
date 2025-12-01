<style>
table { font-size: 0.85em; }
th, td { font-size: 0.9em; padding: 4px 8px; }
</style>

# C09R00 B Axis Half Tesla Pm4Stell.Focus Leaderboard

**Plasma Surface:** `c09r00_B_axis_half_tesla_PM4Stell.focus`

[View all surfaces](../leaderboards/)

---

| # | User | Date | f_B | N | n | avg⟨Bn⟩/⟨B⟩ | κ̄ | F̄ | τ̄ | LN | max(κ) | max(F) | max(τ) | MSC | min(d_cc) | min(d_cs) | L | max⟨Bn⟩/⟨B⟩ | t |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | akaptano | 2025-11-30 | 1.61e-02 | 4 | 4 | 1.47e-01 | 1.51e+00 | 8.96e+04 | 2.66e+04 | 0 | 2.49e+00 | 2.00e+05 | 5.22e+04 | 2.62e+00 | 1.00e-01 | 1.43e-01 | 1.90e+01 | 4.80e-01 | 4.45e+00 |

### Legend

- $f_B = \frac{1}{|S|} \int_{S} \left(\frac{\mathbf{B} \cdot \mathbf{n}}{|\mathbf{B}|}\right)^2 dS$ - Normalized squared flux error on plasma surface
- $N$ - Number of base coils (before applying stellarator symmetry)
- $n$ - Fourier order of coil representation: $\mathbf{r}(\phi) = \sum_{m=-n}^{n} \mathbf{c}_m e^{im\phi}$
- $\frac{1}{|S|} \int_{S} \frac{|\mathbf{B} \cdot \mathbf{n}|}{|\mathbf{B}|} dS$ - Average normalized normal field component
- $\bar{\kappa} = \frac{1}{N} \sum_{i=1}^{N} \kappa_i$ - Mean curvature over all coils, where $\kappa_i = |\mathbf{r}''(s)|$
- $\bar{F} = \frac{1}{N} \sum_{i=1}^{N} \max(|\mathbf{F}_i|)$ - Average of maximum force per coil
- $\bar{\tau} = \frac{1}{N} \sum_{i=1}^{N} \max(|\boldsymbol{\tau}_i|)$ - Average of maximum torque per coil
- $\text{LN} = \frac{1}{4\pi} \sum_{i \neq j} \oint_{C_i} \oint_{C_j} \frac{(\mathbf{r}_i - \mathbf{r}_j) \cdot (d\mathbf{r}_i \times d\mathbf{r}_j)}{|\mathbf{r}_i - \mathbf{r}_j|^3}$ - Linking number between coil pairs
- $\max(\kappa)$ - Maximum curvature across all coils
- $\max(|\mathbf{F}_i|)$ - Maximum force magnitude across all coils
- $\max(|\boldsymbol{\tau}_i|)$ - Maximum torque magnitude across all coils
- $\text{MSC} = \frac{1}{N} \sum_{i=1}^{N} \kappa_i^2$ - Mean squared curvature
- $\min(d_{cc})$ - Minimum coil-to-coil distance
- $\min(d_{cs})$ - Minimum coil-to-surface distance
- $L = \sum_{i=1}^{N} \int_{0}^{L_i} ds$ - Total length of all coils
- $\max\left(\frac{|\mathbf{B} \cdot \mathbf{n}|}{|\mathbf{B}|}\right)$ - Maximum normalized normal field component
- $t$ - Total optimization time (seconds)
