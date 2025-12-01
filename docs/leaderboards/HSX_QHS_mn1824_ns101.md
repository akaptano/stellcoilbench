# Hsx Qhs Mn1824 Ns101 Leaderboard

**Plasma Surface:** `HSX_QHS_mn1824_ns101`

[View all surfaces](../leaderboards/)

---

<table style="font-size: 0.85em;">
<thead>
<tr>
<th style="font-size: 0.9em; padding: 4px 8px;">#</th>
<th style="font-size: 0.9em; padding: 4px 8px;">User</th>
<th style="font-size: 0.9em; padding: 4px 8px;">Date</th>
<th style="font-size: 0.9em; padding: 4px 8px;">f_B</th>
<th style="font-size: 0.9em; padding: 4px 8px;">N</th>
<th style="font-size: 0.9em; padding: 4px 8px;">n</th>
<th style="font-size: 0.9em; padding: 4px 8px;">avg⟨Bn⟩/⟨B⟩</th>
<th style="font-size: 0.9em; padding: 4px 8px;">κ̄</th>
<th style="font-size: 0.9em; padding: 4px 8px;">F̄</th>
<th style="font-size: 0.9em; padding: 4px 8px;">τ̄</th>
<th style="font-size: 0.9em; padding: 4px 8px;">LN</th>
<th style="font-size: 0.9em; padding: 4px 8px;">max(κ)</th>
<th style="font-size: 0.9em; padding: 4px 8px;">max(F)</th>
<th style="font-size: 0.9em; padding: 4px 8px;">max(τ)</th>
<th style="font-size: 0.9em; padding: 4px 8px;">MSC</th>
<th style="font-size: 0.9em; padding: 4px 8px;">min(d_cc)</th>
<th style="font-size: 0.9em; padding: 4px 8px;">min(d_cs)</th>
<th style="font-size: 0.9em; padding: 4px 8px;">L</th>
<th style="font-size: 0.9em; padding: 4px 8px;">max⟨Bn⟩/⟨B⟩</th>
<th style="font-size: 0.9em; padding: 4px 8px;">t</th>
</tr>
</thead>
<tbody>
<tr>
<td style="font-size: 0.9em; padding: 4px 8px;">1</td>
<td style="font-size: 0.9em; padding: 4px 8px;">akaptano</td>
<td style="font-size: 0.9em; padding: 4px 8px;">2025-11-30</td>
<td style="font-size: 0.9em; padding: 4px 8px;">1.75e-02</td>
<td style="font-size: 0.9em; padding: 4px 8px;">4</td>
<td style="font-size: 0.9em; padding: 4px 8px;">4</td>
<td style="font-size: 0.9em; padding: 4px 8px;">1.87e-01</td>
<td style="font-size: 0.9em; padding: 4px 8px;">1.46e+00</td>
<td style="font-size: 0.9em; padding: 4px 8px;">1.92e+07</td>
<td style="font-size: 0.9em; padding: 4px 8px;">8.46e+06</td>
<td style="font-size: 0.9em; padding: 4px 8px;">0</td>
<td style="font-size: 0.9em; padding: 4px 8px;">2.29e+00</td>
<td style="font-size: 0.9em; padding: 4px 8px;">3.61e+07</td>
<td style="font-size: 0.9em; padding: 4px 8px;">1.95e+07</td>
<td style="font-size: 0.9em; padding: 4px 8px;">2.43e+00</td>
<td style="font-size: 0.9em; padding: 4px 8px;">6.15e-03</td>
<td style="font-size: 0.9em; padding: 4px 8px;">4.08e-01</td>
<td style="font-size: 0.9em; padding: 4px 8px;">1.77e+01</td>
<td style="font-size: 0.9em; padding: 4px 8px;">8.73e-01</td>
<td style="font-size: 0.9em; padding: 4px 8px;">1.00e+01</td>
</tr>
</tbody>
</table>

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
