# Landremanpaul2021 Qa Leaderboard

**Plasma Surface:** `LandremanPaul2021_QA`

[View all surfaces](../surfaces.md)

---

| # | User | Date | f_B | N | n | avg⟨Bn⟩/⟨B⟩ | κ̄ | F̄ | τ̄ | LN | max(κ) | max(F) | max(τ) | MSC | min(d_cc) | min(d_cs) | L | max⟨Bn⟩/⟨B⟩ | t |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | akaptano | 2025-11-23 | 9.82e-04 | 4 | 4 | 1.75e-01 | 3.46e+00 | 1.01e+05 | 2.61e+04 | 0 | 1.10e+01 | 1.11e+05 | 3.53e+04 | 2.36e+01 | 1.13e-01 | 1.89e-03 | 1.25e+01 | 4.13e+02 | 8.36e+02 |
| 2 | akaptano@users. | 2025-11-22 | 2.40e-03 | 4 | 16 | 5.52e-02 | 2.81e+00 | 7.37e+06 | 2.36e+06 | 0 | 7.96e+00 | 2.01e+07 | 6.86e+06 | 9.73e+00 | 5.22e-03 | 1.60e-01 | 1.20e+01 | 5.68e-01 | 3.62e+02 |
| 3 | akaptano@users. | 2025-11-22 | 2.40e-03 | 4 | 16 | 5.52e-02 | 2.81e+00 | 7.37e+06 | 2.36e+06 | 0 | 7.96e+00 | 2.01e+07 | 6.86e+06 | 9.73e+00 | 5.22e-03 | 1.60e-01 | 1.20e+01 | 5.68e-01 | 3.27e+02 |
| 4 | akaptano@users. | 2025-11-22 | 2.40e-03 | 4 | 16 | 5.52e-02 | 2.81e+00 | 7.37e+06 | 2.36e+06 | 0 | 7.96e+00 | 2.01e+07 | 6.86e+06 | 9.73e+00 | 5.22e-03 | 1.60e-01 | 1.20e+01 | 5.68e-01 | 3.28e+02 |
| 5 | akaptano | 2025-11-22 | 4.32e-03 | 4 | 16 | 8.40e-02 | 2.98e+00 | 1.97e+05 | 3.74e+04 | 0 | 7.81e+00 | 2.00e+05 | 5.09e+04 | 1.28e+01 | 2.63e-01 | 1.29e-01 | 1.02e+01 | 3.92e-01 | 1.18e+03 |
| 6 | akaptano | 2025-11-23 | 1.03e-02 | 4 | 16 | 1.26e-01 | 2.83e+00 | 1.98e+05 | 3.26e+04 | 0 | 1.08e+01 | 2.18e+05 | 5.52e+04 | 1.12e+01 | 2.78e-01 | 1.25e-01 | 1.09e+01 | 5.55e-01 | 1.25e+03 |
| 7 | akaptano | 2025-11-23 | 1.03e-02 | 4 | 16 | 1.26e-01 | 2.83e+00 | 1.98e+05 | 3.26e+04 | 0 | 1.08e+01 | 2.18e+05 | 5.52e+04 | 1.12e+01 | 2.78e-01 | 1.25e-01 | 1.09e+01 | 5.55e-01 | 1.25e+03 |

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
