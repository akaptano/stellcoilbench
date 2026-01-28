# Circular Tokamak Leaderboard

**Plasma Surface:** `circular_tokamak`

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
<th style="font-size: 0.9em; padding: 4px 8px;">BdotN</th>
<th style="font-size: 0.9em; padding: 4px 8px;">BdotN over B</th>
<th style="font-size: 0.9em; padding: 4px 8px;">arclength variation threshold</th>
<th style="font-size: 0.9em; padding: 4px 8px;">B̄_n</th>
<th style="font-size: 0.9em; padding: 4px 8px;">Var(l_i)</th>
<th style="font-size: 0.9em; padding: 4px 8px;">κ̄</th>
<th style="font-size: 0.9em; padding: 4px 8px;">F̄</th>
<th style="font-size: 0.9em; padding: 4px 8px;">τ̄</th>
<th style="font-size: 0.9em; padding: 4px 8px;">LN</th>
<th style="font-size: 0.9em; padding: 4px 8px;">κ_max</th>
<th style="font-size: 0.9em; padding: 4px 8px;">F_max</th>
<th style="font-size: 0.9em; padding: 4px 8px;">τ_max</th>
<th style="font-size: 0.9em; padding: 4px 8px;">MSC</th>
<th style="font-size: 0.9em; padding: 4px 8px;">d_cc</th>
<th style="font-size: 0.9em; padding: 4px 8px;">d_cs</th>
<th style="font-size: 0.9em; padding: 4px 8px;">L</th>
<th style="font-size: 0.9em; padding: 4px 8px;">max(B_n)</th>
<th style="font-size: 0.9em; padding: 4px 8px;">t</th>
</tr>
</thead>
<tbody>
<tr>
<td style="font-size: 0.9em; padding: 4px 8px;">1</td>
<td style="font-size: 0.9em; padding: 4px 8px;">akaptano</td>
<td style="font-size: 0.9em; padding: 4px 8px;">28/01/26</td>
<td style="font-size: 0.9em; padding: 4px 8px;">7.6e-03</td>
<td style="font-size: 0.9em; padding: 4px 8px;">6</td>
<td style="font-size: 0.9em; padding: 4px 8px;">4</td>
<td style="font-size: 0.9em; padding: 4px 8px;">4.9e-03</td>
<td style="font-size: 0.9em; padding: 4px 8px;">4.5e-03</td>
<td style="font-size: 0.9em; padding: 4px 8px;">0.0e+00</td>
<td style="font-size: 0.9em; padding: 4px 8px;">4.1e-03</td>
<td style="font-size: 0.9em; padding: 4px 8px;">7.0e-03</td>
<td style="font-size: 0.9em; padding: 4px 8px;">2.1e-01</td>
<td style="font-size: 0.9em; padding: 4px 8px;">2.9e+06</td>
<td style="font-size: 0.9em; padding: 4px 8px;">2.4e+05</td>
<td style="font-size: 0.9em; padding: 4px 8px;">0</td>
<td style="font-size: 0.9em; padding: 4px 8px;">2.2e-01</td>
<td style="font-size: 0.9em; padding: 4px 8px;">2.9e+06</td>
<td style="font-size: 0.9em; padding: 4px 8px;">2.5e+05</td>
<td style="font-size: 0.9em; padding: 4px 8px;">4.4e-02</td>
<td style="font-size: 0.9em; padding: 4px 8px;">1.3e+00</td>
<td style="font-size: 0.9em; padding: 4px 8px;">1.6e+00</td>
<td style="font-size: 0.9em; padding: 4px 8px;">1.8e+02</td>
<td style="font-size: 0.9em; padding: 4px 8px;">1.4e-02</td>
<td style="font-size: 0.9em; padding: 4px 8px;">5.6e+01</td>
</tr>
</tbody>
</table>

### Legend

- Normalized squared flux error $f_B = \frac{1}{|S|} \int_{S} \left(\frac{\mathbf{B} \cdot \mathbf{n}}{|\mathbf{B}|}\right)^2 dS$ on plasma surface (dimensionless)
- Number of base coils $N$ (before applying stellarator symmetry) (dimensionless)
- Fourier order $n$ of coil representation: $\mathbf{r}(\phi) = \mathbf{a}_0 + \sum_{m=1}^{n} \left[\mathbf{a}_m \cos(m\phi) + \mathbf{b}_m \sin(m\phi)\right]$ (dimensionless)
- Bdotn
- Bdotn Over B
- Arclength Variation Threshold
- Average normalized normal field component $\bar{B}_n$ where $B_n = \frac{|\mathbf{B} \cdot \mathbf{n}|}{|\mathbf{B}|}$ and $\bar{B}_n = \frac{\int_{S} |\mathbf{B} \cdot \mathbf{n}| dS}{\int_{S} |\mathbf{B}| dS}$ (dimensionless)
- Variance of incremental arclength $J = \text{Var}(l_i)$ where $l_i$ is the average incremental arclength on interval $I_i$ from a partition $\{I_i\}_{i=1}^L$ of $[0,1]$ ($\text{m}^2$)
- Mean curvature $\bar{\kappa} = \frac{1}{N} \sum_{i=1}^{N} \kappa_i$ over all coils, where $\kappa_i = |\mathbf{r}''(s)|$ ($\text{m}^{-1}$)
- Average of maximum force $\bar{F} = \frac{1}{N} \sum_{i=1}^{N} \max(|\mathbf{F}_i|)$ per coil ($\text{N}/\text{m}$)
- Average of maximum torque $\bar{\tau} = \frac{1}{N} \sum_{i=1}^{N} \max(|\boldsymbol{\tau}_i|)$ per coil ($\text{N}$)
- Linking number $\text{LN} = \frac{1}{4\pi} \sum_{i \neq j} \oint_{C_i} \oint_{C_j} \frac{(\mathbf{r}_i - \mathbf{r}_j) \cdot (d\mathbf{r}_i \times d\mathbf{r}_j)}{|\mathbf{r}_i - \mathbf{r}_j|^3}$ between coil pairs (dimensionless)
- Maximum curvature $\kappa_\text{max}$ across all coils ($\text{m}^{-1}$)
- Maximum force magnitude $F_\text{max}$ across all coils ($\text{N}/\text{m}$)
- Maximum torque magnitude $\tau_\text{max}$ across all coils ($\text{N}$)
- Mean squared curvature $\text{MSC} = \frac{1}{N} \sum_{i=1}^{N} \kappa_i^2$ ($\text{m}^{-2}$)
- Minimum coil-to-coil distance $d_{cc}$ ($\text{m}$)
- Minimum coil-to-surface distance $d_{cs}$ ($\text{m}$)
- Total length $L = \sum_{i=1}^{N} \int_{0}^{L_i} ds$ of all coils ($\text{m}$)
- Maximum normalized normal field component $\max(B_n)$ where $B_n = \frac{|\mathbf{B} \cdot \mathbf{n}|}{|\mathbf{B}|}$ (dimensionless)
- Total optimization time $t$ ($\text{s}$)
