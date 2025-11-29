# Hsx Qhs Mn1824 Ns101 Leaderboard

**Plasma Surface:** `HSX_QHS_mn1824_ns101`

[View all surfaces](../leaderboards/)

---

| # | User | Date | f_B | N | n | avg⟨Bn⟩/⟨B⟩ | κ̄ | F̄ | τ̄ | LN | max(κ) | max(F) | max(τ) | MSC | min(d_cc) | min(d_cs) | L | max⟨Bn⟩/⟨B⟩ | t |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | akaptano | 2025-11-29 | 1.75e-02 | 4 | 4 | 1.87e-01 | 1.46e+00 | 1.92e+07 | 8.46e+06 | 0 | 2.29e+00 | 3.61e+07 | 1.95e+07 | 2.43e+00 | 6.15e-03 | 4.08e-01 | 1.77e+01 | 8.73e-01 | 1.00e+01 |

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
