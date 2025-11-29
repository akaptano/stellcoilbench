# Landremanpaul2021 Qa Leaderboard

**Plasma Surface:** `LandremanPaul2021_QA`

[View all surfaces](../leaderboards/)

---

| # | User | Date | f_B | N | n | avg⟨Bn⟩/⟨B⟩ | κ̄ | F̄ | τ̄ | LN | max(κ) | max(F) | max(τ) | MSC | min(d_cc) | min(d_cs) | L | max⟨Bn⟩/⟨B⟩ | t |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | akaptano | 2025-11-28 | 4.72e-05 | 4 | 8 | 5.51e-02 | 3.16e+00 | 6.87e+04 | 2.36e+04 | 0 | 1.26e+01 | 7.69e+04 | 3.34e+04 | 1.77e+01 | 7.49e-02 | 3.07e-04 | 1.47e+01 | 9.98e-01 | 1.89e+03 |
| 2 | akaptano | 2025-11-29 | 7.55e-04 | 4 | 8 | 1.09e-01 | 3.28e+00 | 9.51e+04 | 2.13e+04 | 0 | 1.12e+01 | 1.09e+05 | 3.95e+04 | 1.78e+01 | 1.24e-01 | 9.80e-04 | 1.17e+01 | 9.96e-01 | 7.52e+02 |
| 3 | akaptano | 2025-11-29 | 7.55e-04 | 4 | 8 | 1.09e-01 | 3.28e+00 | 9.51e+04 | 2.13e+04 | 0 | 1.12e+01 | 1.09e+05 | 3.95e+04 | 1.78e+01 | 1.24e-01 | 9.80e-04 | 1.17e+01 | 9.96e-01 | 7.49e+02 |
| 4 | akaptano | 2025-11-29 | 7.55e-04 | 4 | 8 | 1.09e-01 | 3.28e+00 | 9.51e+04 | 2.13e+04 | 0 | 1.12e+01 | 1.09e+05 | 3.95e+04 | 1.78e+01 | 1.24e-01 | 9.80e-04 | 1.17e+01 | 9.96e-01 | 7.55e+02 |
| 5 | akaptano | 2025-11-28 | 7.55e-04 | 4 | 8 | 1.09e-01 | 3.28e+00 | 9.51e+04 | 2.13e+04 | 0 | 1.12e+01 | 1.09e+05 | 3.95e+04 | 1.78e+01 | 1.24e-01 | 9.80e-04 | 1.17e+01 | 9.96e-01 | 7.66e+02 |
| 6 | akaptano | 2025-11-28 | 1.63e-02 | 4 | 4 | 1.59e-01 | 2.65e+00 | 2.35e+05 | 4.16e+04 | 0 | 7.49e+00 | 2.45e+05 | 4.97e+04 | 1.07e+01 | 2.19e-01 | 1.06e-01 | 1.08e+01 | 4.71e-01 | 3.15e+01 |
| 7 | akaptano | 2025-11-29 | 1.63e-02 | 4 | 4 | 1.59e-01 | 2.65e+00 | 2.35e+05 | 4.16e+04 | 0 | 7.49e+00 | 2.45e+05 | 4.97e+04 | 1.07e+01 | 2.19e-01 | 1.06e-01 | 1.08e+01 | 4.71e-01 | 3.20e+01 |
| 8 | akaptano | 2025-11-28 | 1.63e-02 | 4 | 4 | 1.59e-01 | 2.65e+00 | 2.35e+05 | 4.16e+04 | 0 | 7.49e+00 | 2.45e+05 | 4.97e+04 | 1.07e+01 | 2.19e-01 | 1.06e-01 | 1.08e+01 | 4.71e-01 | 3.22e+01 |

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
