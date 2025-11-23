# Muse.Focus Leaderboard

**Plasma Surface:** `MUSE.focus`

[View all surfaces](../surfaces.md)

---

| # | User | Date | f_B | N | n | avg⟨Bn⟩/⟨B⟩ | κ̄ | F̄ | τ̄ | LN | max(κ) | max(F) | max(τ) | MSC | min(d_cc) | min(d_cs) | L | max⟨Bn⟩/⟨B⟩ | t |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | akaptano | 2025-11-23 | 1.16e-04 | 4 | 4 | 1.30e-02 | 5.61e+00 | 2.67e+03 | 4.99e+02 | 12 | 8.34e+00 | 4.99e+03 | 1.17e+03 | 3.29e+01 | 2.21e-02 | 6.68e-02 | 4.85e+00 | 5.95e-02 | 7.07e+02 |
| 2 | akaptano | 2025-11-23 | 1.18e-02 | 4 | 16 | 1.24e-01 | 2.89e+00 | 6.09e+05 | 4.12e+04 | 2 | 3.33e+00 | 6.53e+05 | 8.74e+04 | 8.35e+00 | 5.15e-04 | 1.84e-01 | 8.71e+00 | 4.54e-01 | 3.43e+02 |
| 3 | akaptano | 2025-11-23 | 1.18e-02 | 4 | 16 | 1.24e-01 | 2.89e+00 | 6.09e+05 | 4.12e+04 | 2 | 3.33e+00 | 6.53e+05 | 8.74e+04 | 8.35e+00 | 5.15e-04 | 1.84e-01 | 8.71e+00 | 4.54e-01 | 3.45e+02 |

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
