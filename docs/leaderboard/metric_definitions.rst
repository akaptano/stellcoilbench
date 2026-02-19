Metric Definitions
===================

The following metrics are used to evaluate coil optimization submissions:

Notation
--------

The following notation is used throughout the mathematical definitions:

- :math:`C_i` denotes coil curve :math:`i`
- :math:`S` denotes the plasma surface
- :math:`\mathbf{r}_i` denotes a point on coil curve :math:`C_i`
- :math:`\mathbf{s}` denotes a point on the plasma surface :math:`S`
- :math:`\ell_i` denotes arclength along coil curve :math:`C_i`
- :math:`L_i` denotes the total length of coil curve :math:`C_i`
- :math:`\kappa_i` denotes curvature along coil curve :math:`C_i`
- :math:`\frac{d\vec{F}_i}{d\ell_i}` denotes force per unit length on coil curve :math:`C_i`
- :math:`\frac{d\vec{T}_i}{d\ell_i}` denotes torque per unit length on coil curve :math:`C_i`
- :math:`N` denotes the number of coils
- :math:`d\ell_i` denotes the differential arclength element along coil curve :math:`C_i`
- :math:`ds` denotes the differential surface area element on the plasma surface :math:`S`
- :math:`\mathbf{B}` denotes the magnetic field vector
- :math:`\mathbf{n}` denotes the unit normal vector to the plasma surface

Field Quality Metrics
---------------------

**Average Normalized Normal Field Component** (:math:`\bar{B}_n`)
   Average of the absolute value of the normalized normal field component across the plasma surface.
   
   Mathematical form:
   
   .. math::
      B_n = \frac{|\mathbf{B} \cdot \mathbf{n}|}{|\mathbf{B}|}
   
   .. math::
      \bar{B}_n = \frac{\int_{S} |\mathbf{B} \cdot \mathbf{n}| ds}{\int_{S} |\mathbf{B}| ds}
   
   Units: dimensionless
   
   Lower values indicate better field quality.

**Maximum Normalized Normal Field Component** (:math:`\max(B_n)`)
   Maximum value of the normalized normal field component across the plasma surface.
   
   Mathematical form:
   
   .. math::
      B_n = \frac{|\mathbf{B} \cdot \mathbf{n}|}{|\mathbf{B}|}
   
   .. math::
      \max(B_n) = \max_{\mathbf{s} \in S} B_n(\mathbf{s})
   
   Units: dimensionless
   
   Lower values indicate better field quality.

Coil Geometry Metrics
---------------------

**Fourier Order** (:math:`n`)
   Order of the Fourier series representation used for coil curves.
   
   Mathematical form:
   
   .. math::
      \mathbf{r}(\phi) = \mathbf{a}_0 + \sum_{m=1}^{n} \left[\mathbf{a}_m \cos(m\phi) + \mathbf{b}_m \sin(m\phi)\right]
   
   where :math:`\mathbf{a}_0`, :math:`\mathbf{a}_m`, and :math:`\mathbf{b}_m` are Fourier coefficients and :math:`\phi` is the parameterization angle.
   
   Units: dimensionless
   
   Higher orders allow more complex coil shapes but increase the number of optimization variables.

**Arclength Variation** (:math:`J`)
   Variance of incremental arclength between coil segments.
   
   Mathematical form:
   
   .. math::
      J = \text{Var}(l_i)
   
   where :math:`l_i` is the average incremental arclength on interval :math:`I_i` from a partition :math:`\{I_i\}_{i=1}^L` of :math:`[0,1]`.
   
   Units: :math:`\text{m}^2` (meters squared)
   
   Lower values indicate more uniform spacing along coils, which is important for manufacturing and field quality.

**Mean Curvature** (:math:`\bar{\kappa}`)
   Average curvature across all coils.
   
   Mathematical form:
   
   .. math::
      \kappa_i(\ell_i) = \left|\mathbf{r}_i''(\ell_i)\right|
   
   .. math::
      \bar{\kappa} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{L_i} \int_{C_i} \kappa_i(\ell_i) ~d\ell_i
   
   where :math:`\mathbf{r}_i(\ell_i)` is the parameterization of coil curve :math:`C_i` by arclength.
   
   Units: :math:`\text{m}^{-1}` (inverse meters)
   
   Lower curvature values indicate smoother coils that are easier to manufacture.

**Maximum Curvature** (:math:`\kappa_\text{max}`)
   Maximum curvature value across all coils.
   
   Mathematical form:
   
   .. math::
      \kappa_\text{max} = \max_{i=1,\ldots,N} \max_{\ell_i \in [0,L_i]} \kappa_i(\ell_i)
   
   Units: :math:`\text{m}^{-1}` (inverse meters)
   
   Lower values indicate coils without extreme curvature regions.

**Mean Squared Curvature** (:math:`\text{MSC}`)
   Mean squared curvature per coil, averaged across all coils.
   
   Mathematical form:
   
   .. math::
      J = \frac{1}{L_i} \int_{C_i} \kappa_i^2(\ell_i) ~d\ell_i
   
   .. math::
      \text{MSC} = \frac{1}{N} \sum_{i=1}^{N} J_i
   
   where :math:`L_i` is the total length of coil curve :math:`C_i`, :math:`\ell_i` is the arclength along the curve, and :math:`\kappa_i` is the curvature.
   
   Units: :math:`\text{m}^{-2}` (inverse meters squared)
   
   This provides a smoother penalty than maximum curvature, encouraging overall smoothness rather than just avoiding extreme values.

**Total Length** (:math:`L`)
   Total length of all coils.
   
   Mathematical form:
   
   .. math::
      L = \sum_{i=1}^{N} \int_{C_i} d\ell_i
   
   Units: :math:`\text{m}` (meters)
   
   Shorter coils are generally preferred for reduced material costs and improved manufacturability.

**Fourier Continuation (FC)**
   Sequence of Fourier orders used in continuation method. The optimization starts with a low-order representation, converges, then extends the solution to higher orders using the previous solution as initial condition. This helps achieve convergence for complex problems.
   
   Format: comma-separated list of orders (e.g., "4,6,8" means optimization was performed at orders 4, 6, and 8 sequentially). If not used, the column shows "—".

**Number of Base Coils** (:math:`N`)
   Number of base coils before applying stellarator symmetry.
   
   Units: dimensionless
   
   Typical values: 4, 6, 8, 12. More coils allow more complex field shaping but increase computational cost.

**Total Superconductor Length** (:math:`L_{\text{SC}}`)
   Total length of superconducting tape required at reactor scale, accounting for the number of turns in each coil's winding pack.
   
   Mathematical form:
   
   .. math::
      L_{\text{SC}} = \frac{1}{1000} \sum_{i=1}^{N_{\text{coils}}} N_{\text{turns},i} \times L_{\text{reactor},i}
   
   where :math:`N_{\text{turns},i} = \max(N_{F,i},\, N_{J_c,i})` is the number of turns per coil (driven by force limits or REBCO critical-current limits, whichever is larger), and :math:`L_{\text{reactor},i}` is the reactor-scale length of coil :math:`i`. The factor of 1/1000 converts from meters to kilometers.
   
   Units: :math:`\text{km}` (kilometers)
   
   Lower values indicate more economical coil designs requiring less superconducting material. This is a derived reactor-scale metric that combines the winding-pack turn count with the scaled coil lengths.

Separation Metrics
------------------

**Minimum Coil-to-Coil Distance** (:math:`d_{cc}`)
   Minimum distance between any two coils.
   
   Mathematical form:
   
   .. math::
      d_{cc} = \min_{i \neq j} \min_{\mathbf{r}_i \in C_i, \mathbf{r}_j \in C_j} \left\| \mathbf{r}_i - \mathbf{r}_j \right\|_2
   
   Units: :math:`\text{m}` (meters)
   
   Ensures coils maintain a safe separation distance to prevent collisions.

**Minimum Coil-to-Surface Distance** (:math:`d_{cs}`)
   Minimum distance between any coil and the plasma surface.
   
   Mathematical form:
   
   .. math::
      d_{cs} = \min_{i} \min_{\mathbf{r}_i \in C_i, \mathbf{s} \in S} \left\| \mathbf{r}_i - \mathbf{s} \right\|_2
   
   Units: :math:`\text{m}` (meters)
   
   Ensures coils maintain a safe distance from the plasma surface.

Force and Torque Metrics
------------------------

**Average of Maximum Force** (:math:`\bar{F}`)
   Average across coils of the maximum force magnitude per coil.
   
   Mathematical form:
   
   .. math::
      \bar{F} = \frac{1}{N} \sum_{i=1}^{N} \max_{\ell_i \in [0,L_i]} \left|\frac{d\vec{F}_i}{d\ell_i}\right|
   
   where :math:`\frac{d\vec{F}_i}{d\ell_i}` is the Lorentz force per unit length on coil curve :math:`C_i`.
   
   Units: :math:`\text{N}/\text{m}` (Newtons per meter)
   
   Lower values indicate coils that are easier to support mechanically.

**Average of Maximum Torque** (:math:`\bar{\tau}`)
   Average across coils of the maximum torque magnitude per coil.
   
   Mathematical form:
   
   .. math::
      \bar{\tau} = \frac{1}{N} \sum_{i=1}^{N} \max_{\ell_i \in [0,L_i]} \left|\frac{d\vec{T}_i}{d\ell_i}\right|
   
   where :math:`\frac{d\vec{T}_i}{d\ell_i}` is the Lorentz torque per unit length on coil curve :math:`C_i`.
   
   Units: :math:`\text{N}` (Newtons)
   
   Lower values indicate coils with reduced rotational forces that must be resisted by supports.

**Maximum Force Magnitude** (:math:`F_\text{max}`)
   Maximum force magnitude across all coils.
   
   Mathematical form:
   
   .. math::
      F_\text{max} = \max_{i=1,\ldots,N} \max_{\ell_i \in [0,L_i]} \left|\frac{d\vec{F}_i}{d\ell_i}\right|
   
   Units: :math:`\text{N}/\text{m}` (Newtons per meter)
   
   High forces indicate coils that may be difficult to support mechanically.

**Maximum Torque Magnitude** (:math:`\tau_\text{max}`)
   Maximum torque magnitude across all coils.
   
   Mathematical form:
   
   .. math::
      \tau_\text{max} = \max_{i=1,\ldots,N} \max_{\ell_i \in [0,L_i]} \left|\frac{d\vec{T}_i}{d\ell_i}\right|
   
   Units: :math:`\text{N}` (Newtons)
   
   High torques can lead to mechanical instability.

Topology Metrics
----------------

**Linking Number** (:math:`\text{LN}`)
   Topological measure of how coils are linked together.
   
   Mathematical form:
   
   .. math::
      \text{LN} = \frac{1}{4\pi} \sum_{i \neq j} \oint_{C_i} \oint_{C_j} \frac{\left(\mathbf{r}_i - \mathbf{r}_j\right) \cdot \left(d\mathbf{r}_i \times d\mathbf{r}_j\right)}{\left|\mathbf{r}_i - \mathbf{r}_j\right|^3}
   
   Units: dimensionless
   
   This metric ensures coils maintain their topological structure during optimization.

Performance Metrics
-------------------

**Total Optimization Time** (:math:`t`)
   Total time required to complete the optimization.
   
   Units: :math:`\text{s}` (seconds)
   
   Lower values indicate more efficient optimization algorithms or faster convergence.

Particle Confinement Metrics
----------------------------

**Loss Fraction** (:math:`\text{LF}`)
   Final particle loss fraction from SIMPLE fast particle tracing.
   
   Mathematical form:
   
   .. math::
      \text{LF} = 1 - f_c
   
   where :math:`f_c` is the confined fraction (sum of confined passing and trapped particles).
   
   Units: dimensionless
   
   Lower values indicate better particle confinement. A value of 0 means all particles are confined, while a value of 1 means all particles are lost. This metric is computed by the SIMPLE code using Monte Carlo particle tracing.

**Average Quasisymmetry Error** (:math:`\text{avg}(QS)`)
   Average two-term quasisymmetry error computed from VMEC equilibrium.
   
   Mathematical form:
   
   .. math::
      QS = \frac{|\mathbf{B}|_{m,n}}{|\mathbf{B}|}
   
   The two-term quasisymmetry error measures how well the magnetic field strength :math:`|\mathbf{B}|` is constant on flux surfaces by evaluating the ratio residual where :math:`(m,n)` is the target helicity.
   
   Units: dimensionless
   
   Lower values indicate better quasisymmetry, which is important for particle confinement in stellarators.

Composite Score
---------------

The **Score** column is a single-number summary of reactor-scale
engineering feasibility.  It is computed in two stages.

Stage 1: Hard Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~

Hard constraints are evaluated first.  If **any** hard constraint is
violated the score is set to **0** and the entry is marked **FAIL**
(excluded from the main leaderboard).  Missing metrics are skipped.

Two hard constraints apply a transform before comparison:

- **Coil-coil linking number** — compared as :math:`|\text{LN}|`
  (absolute value), since the linking number may be negative.
- **Max turns per coil** — the element-wise maximum of the per-coil
  turn-count list is compared against the bound.

Stage 2: Soft Constraint Geometric Mean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each soft constraint whose metric is available (and whose bound is
non-zero), a margin exponent :math:`m_i` is computed:

.. math::

   m_i = \begin{cases}
     1 - \text{value}_i\,/\,\text{bound}_i
       & \text{value} \leq \text{bound} \;(\text{upper-bound constraint})\\[6pt]
     \text{value}_i\,/\,\text{bound}_i - 1
       & \text{value} \geq \text{bound} \;(\text{lower-bound constraint})
   \end{cases}

Positive :math:`m_i` = constraint satisfied with margin; negative =
violated.

Two soft constraints apply a transform to the raw metric before the
margin calculation:

- **RMS curvature** — :math:`\sqrt{\text{MSC}}` is used (square root
  of the mean-squared curvature) so the comparison is in m\ :sup:`-1`.
- **Arclength variation** — :math:`\sqrt{\text{Var}}` is used (square
  root of the variance) so the comparison is in m.

The composite score is the geometric mean of the individual exponential
factors:

.. math::

   \text{Score}     = \exp\!\left(\frac{1}{n}\sum_{i=1}^{n} m_i\right)     = \left(\prod_{i=1}^{n} e^{m_i}\right)^{\!1/n}

where :math:`n` is the number of soft constraints for which metrics
were available.

Soft Constraints and Margin Formulas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following table lists every soft constraint that enters the
composite score, together with its reactor-scale bound, the source
metric key, and the resulting margin formula.

.. list-table::
   :header-rows: 1
   :widths: 28 12 13 47

   * - Constraint
     - Direction
     - Bound
     - Margin :math:`m_i`
   * - avg :math:`\langle B{\cdot}n\rangle / \langle B\rangle` (``avg_BdotN_over_B``)
     - :math:`\leq`
     - 0.01
     - :math:`1 - \text{value}\;/\;0.01`
   * - Min coil-surface distance (``reactor_scale_min_cs_separation``)
     - :math:`\geq`
     - 1.3 m
     - :math:`\text{value}\;/\;1.3 - 1`
   * - Min coil-coil distance (``reactor_scale_min_cc_separation``)
     - :math:`\geq`
     - 0.7 m
     - :math:`\text{value}\;/\;0.7 - 1`
   * - Total coil length (``reactor_scale_total_length``)
     - :math:`\leq`
     - 220 m
     - :math:`1 - L\;/\;220`
   * - Max curvature :math:`\kappa` (``reactor_scale_max_curvature``)
     - :math:`\leq`
     - 1.0 m\ :sup:`-1`
     - :math:`1 - \kappa_{\max}\;/\;1.0`
   * - RMS curvature :math:`\sqrt{\text{MSC}}` (``reactor_scale_mean_squared_curvature``)
     - :math:`\leq`
     - 1.0 m\ :sup:`-1`
     - :math:`1 - \sqrt{\text{MSC}}\;/\;1.0`
   * - Arclength variation :math:`\sqrt{\text{Var}}` (``reactor_scale_arclength_variation``)
     - :math:`\leq`
     - 1.0 m
     - :math:`1 - \sqrt{\text{Var}}\;/\;1.0`
   * - Total superconductor length :math:`L_{\text{SC}}` (``total_superconductor_length_km``)
     - :math:`\leq`
     - 100 km
     - :math:`1 - L_{\text{SC}}\;/\;100`

Interpretation
~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Score = 0**
     - Hard infeasibility (coils delinked from plasma, coils interlinked,
       winding packs overlap, or too many turns required).
   * - **0 < Score < 1**
     - Feasible but one or more soft constraints violated on average.
   * - **Score = 1**
     - All soft constraints met exactly on average.
   * - **Score > 1**
     - All constraints satisfied with engineering margin.
   * - **Score = None**
     - No soft-constraint metrics available (legacy or missing data).

Entries are sorted by composite score **descending** (higher is better).
Legacy entries without a composite score fall back to
``score_primary`` (lower squared flux is better) and appear after
scored entries.

Worked Example
~~~~~~~~~~~~~~

Consider a reactor-scale design with:

- avg :math:`B_n/|B|` = 0.005, :math:`d_{cs}` = 1.5 m,
  :math:`d_{cc}` = 0.8 m, :math:`L` = 200 m,
  :math:`\kappa_{\max}` = 0.7, :math:`\sqrt{\text{MSC}}` = 0.6,
  :math:`\sqrt{\text{Var}}` = 0.4, :math:`L_{\text{SC}}` = 50 km

.. math::

   m_1 &= 1 - 0.005/0.01   &&= 0.500 \\
   m_2 &= 1.5/1.3 - 1      &&\approx 0.154 \\
   m_3 &= 0.8/0.7 - 1      &&\approx 0.143 \\
   m_4 &= 1 - 200/220       &&\approx 0.091 \\
   m_5 &= 1 - 0.7/1.0       &&= 0.300 \\
   m_6 &= 1 - 0.6/1.0       &&= 0.400 \\
   m_7 &= 1 - 0.4/1.0       &&= 0.600 \\
   m_8 &= 1 - 50/100         &&= 0.500

Mean margin :math:`= (0.500 + 0.154 + 0.143 + 0.091 + 0.300 + 0.400 + 0.600 + 0.500)\,/\,8 \approx 0.336`.

Score :math:`= \exp(0.336) \approx` **1.399** — all constraints
satisfied with comfortable engineering margin.

Reactor-Scale Constraints
-------------------------

All submissions are scaled to the ARIES-CS reference reactor
(minor radius :math:`a = 1.7\,\text{m}`, on-axis field
:math:`B_0 = 5.7\,\text{T}`) before engineering feasibility is assessed.

**Hard feasibility constraints** — any violation makes the design infeasible
(score = 0, excluded from the main leaderboard):

.. list-table::
   :header-rows: 1

   * - Constraint
     - Bound
     - Description
   * - Coils linked to plasma surface
     - = True
     - Every base coil must topologically encircle the plasma.
   * - Coil-coil linking number (\|LN\| ≈ 0)
     - ≤ 0.5 (dimensionless)
     - Coils must not interlink with one another.
   * - Max turns per coil (N_turns ≤ 500)
     - ≤ 500
     - With :math:`N_{\text{turns}}` chosen to keep per-turn force ≤ 0.5 MN/m, no coil may require more than 500 turns.
   * - Finite-build coil-coil clearance (d_cc > w_WP)
     - ≥ 0.0 m
     - Centreline distance :math:`d_{\text{cc,min}}` must exceed the largest winding-pack width :math:`w_{\text{WP,max}}` to prevent physical overlap of finite-build coils.

**Soft engineering constraints** — contribute to the composite score via
exponential margin factors.  Violations lower the score below 1 but do not
set it to zero:

.. list-table::
   :header-rows: 1

   * - Metric
     - Bound
     - Direction
     - Units
   * - avg ⟨B·n⟩/⟨B⟩
     - :math:`\leq 0.01`
     - max
     - (dimensionless)
   * - Minimum coil-surface distance
     - :math:`\geq 1.3`
     - min
     - m
   * - Minimum coil-coil distance
     - :math:`\geq 0.7`
     - min
     - m
   * - Total coil length
     - :math:`\leq 220.0`
     - max
     - m
   * - Max curvature κ
     - :math:`\leq 1.0`
     - max
     - m⁻¹
   * - Max √MSC (RMS curvature)
     - :math:`\leq 1.0`
     - max
     - m⁻¹
   * - Arclength variation √Var
     - :math:`\leq 1.0`
     - max
     - m
   * - Total superconductor length L_SC
     - :math:`\leq 100.0`
     - max
     - km

Winding-Pack Turn-Count Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The *simsopt* optimiser models each coil as a single filamentary turn
carrying the total current :math:`I`.  In a real reactor the winding pack
contains :math:`N_{\text{turns}}` turns, each carrying :math:`I / N_{\text{turns}}`.
We estimate the required number of turns from **two independent criteria**
and take the element-wise maximum:

.. math::

   N_{\text{turns},\,i}     = \max\!\bigl(N_{\text{turns},\,i}^{(\text{force})},\;                    N_{\text{turns},\,i}^{(J_c)}\bigr)

1. Force-based turns
^^^^^^^^^^^^^^^^^^^^

With :math:`N` turns the Lorentz force per unit length on each turn is

.. math::

   F_{\text{turn}} = \frac{F_{\text{reactor,single-turn}}}{N_{\text{turns}}}

For each coil we find the minimum :math:`N` to keep
:math:`F_{\text{turn}} \leq 0.5\,\text{MN/m}`:

.. math::

   N_{\text{turns},\,i}^{(\text{force})}     = \left\lceil \frac{F_{\text{reactor},\,i}}{0.5\;\text{MN/m}} \right\rceil

2. Critical-current-density-based turns
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This criterion ensures the HTS superconductor operates within its critical
envelope.  The model follows the Stellaris winding-pack design
(Lion *et al.*, *Fusion Engineering and Design* **214**, 2025, 114868,
Table 7–8 and Section 2.9).

**REBCO tape-stack** :math:`J_c` **model.**
A Kim-like parametrisation calibrated to tape-stack data at 20 K
(field-aligned tapes):

.. math::

   J_c(B, T) = \frac{C_0}{1 + (B/B_0)^\alpha}     \;\times\;\frac{1 - T/T_c}{1 - T_{\text{ref}}/T_c}

with fitted constants at :math:`T_{\text{ref}} = 20\,\text{K}`:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Value
     - Description
   * - :math:`C_0`
     - :math:`5.0 \times 10^{9}\;\text{A/m}^2`
     - Zero-field engineering :math:`J_c` (≈ 5000 A/mm²)
   * - :math:`B_0`
     - 18.14 T
     - Characteristic field
   * - :math:`\alpha`
     - 0.902
     - Field exponent
   * - :math:`T_c`
     - 92 K
     - REBCO critical temperature

Validation against Stellaris Table 8:
:math:`B = 20\,\text{T} \rightarrow J_c \approx 2450\;\text{A/mm}^2`,
:math:`B = 25\,\text{T} \rightarrow J_c \approx 2200\;\text{A/mm}^2`.

**Stellaris winding-pack parameters.**

.. list-table::
   :header-rows: 1

   * - Parameter
     - Value
     - Description
   * - :math:`T_{\text{op}}`
     - 20 K
     - Operating temperature
   * - :math:`\eta`
     - 0.80
     - Utilisation cap (:math:`J_{\text{op}} / J_c \leq \eta`)
   * - :math:`I_{\text{lead,max}}`
     - 50 kA
     - Current-lead limit
   * - :math:`A_{\text{HTS}}`
     - :math:`36\;\text{mm}^2` (6 mm × 6 mm)
     - HTS tape-stack cross-section per turn
   * - :math:`A_{\text{turn}}`
     - :math:`400\;\text{mm}^2` (20 mm × 20 mm)
     - Total turn cross-section (incl. stabiliser, insulation, structure)
   * - :math:`f_{\text{WP}}`
     - 1.3
     - Winding-pack self-field enhancement factor

**Algorithm for each coil** :math:`i`:

1. **Required ampere-turns** at reactor scale:

   .. math::

      NI_i = I_{\text{device},i} \times B_{\text{scale}} \times L_{\text{scale}}

   where :math:`I_{\text{device},i}` is the *simsopt* single-turn current.
   If per-coil currents are unavailable, :math:`I` is estimated from
   the force data: :math:`I \approx (F/L) / B_{\text{device}}`.

2. **Peak conductor field** estimate:

   .. math::

      B_{\text{ext},i} = \frac{(F/L)_{\text{device},i}}{I_{\text{device},i}} \times B_{\text{scale}},      \qquad      B_{\text{peak},i} = f_{\text{WP}} \times B_{\text{ext},i}

   The factor :math:`f_{\text{WP}} = 1.3` accounts for the additional
   self-field produced by the multi-turn winding pack at its inner edge.

3. **Critical current of the HTS cable**:

   .. math::

      I_{c,\text{cable}} = J_c(B_{\text{peak}},\; T_{\text{op}}) \times A_{\text{HTS}}

4. **Operating current per turn** (lead- or tape-limited):

   .. math::

      I_{\text{turn}} = \min\!\bigl(I_{\text{lead,max}},\; \eta \times I_{c,\text{cable}}\bigr)

5. **Number of turns** from :math:`J_c` requirements:

   .. math::

      N_{\text{turns},\,i}^{(J_c)} = \left\lceil \frac{NI_i}{I_{\text{turn},i}} \right\rceil

**Hard constraint.**
The final :math:`N_{\text{turns},i}` (element-wise maximum of force and
:math:`J_c` requirements) must satisfy
:math:`\max_i N_{\text{turns},\,i} \leq 500`.

3. Finite-build (winding-pack) extent
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With each turn occupying :math:`A_{\text{turn}} = 20\;\text{mm} \times
20\;\text{mm} = 400\;\text{mm}^2` (Table 7 of Lion *et al.*: this area
includes the REBCO tape stack, copper stabiliser, solder, steel jacket,
and helium cooling channel), a square winding pack with :math:`N` turns
has side length

.. math::

   w_{\text{WP}} = \sqrt{N_{\text{turns}}} \times 20\;\text{mm}

Validation against Stellaris Table 8:

- Coil 0: :math:`N = 324 \;\Rightarrow\; w = 18 \times 20\;\text{mm} = 360\;\text{mm} \;\checkmark`
- Coil 5: :math:`N = 225 \;\Rightarrow\; w = 15 \times 20\;\text{mm} = 300\;\text{mm} \;\checkmark`

The leaderboard reports :math:`w_{\text{WP}}` — the **maximum** winding-pack
side length across all coils (in metres).  This gives the finite-build extent
that must be accommodated by the coil-surface and coil-coil separation gaps.

4. Finite-build coil-coil intersection check
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*simsopt*'s ``CurveCurveDistance`` penalty measures the **centreline-to-centreline**
distance between coil filaments.  Once the winding-pack extent is known, we
can check whether the finite-build coils would physically overlap.

Each coil's winding pack extends :math:`w_i / 2` from the centreline in every
direction.  For two coils *i* and *j* separated by centreline distance
:math:`d_{ij}`, the clearance between their outer edges is

.. math::

   \text{clearance}_{ij} = d_{ij} - \frac{w_i}{2} - \frac{w_j}{2}

Because we only store the **global minimum** coil-coil distance
:math:`d_{\text{cc,min}} = \min_{i<j} d_{ij}`, the most conservative
check uses the largest winding-pack width for both coils:

.. math::

   \text{clearance} = d_{\text{cc,min}} - w_{\text{WP,max}}

where :math:`w_{\text{WP,max}} = \max_i w_{\text{WP},i}`.  This is a **hard
constraint**: if the clearance is negative (:math:`d_{\text{cc,min}} <
w_{\text{WP,max}}`), the winding packs would intersect and the design is
infeasible (score = 0).

5. Per-turn force and torque
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once :math:`N_{\text{turns},i}` is known for each coil, the engineering-relevant
structural loads are the per-turn quantities:

.. math::

   F_{\text{turn},i} = \frac{F_{\text{reactor},i}}{N_{\text{turns},i}},   \qquad   \tau_{\text{turn},i} = \frac{\tau_{\text{reactor},i}}{N_{\text{turns},i}}

The leaderboard reports:

- :math:`F_{\text{turn}}` — :math:`\max_i F_{\text{turn},i}` (MN/m), the
  maximum per-turn force across all coils.
- :math:`\tau_{\text{turn}}` — :math:`\max_i \tau_{\text{turn},i}` (MN), the
  maximum per-turn torque across all coils.

These replace the single-turn :math:`F_{\max}` and :math:`\tau_{\max}` in the
reactor-scale leaderboard, since the single-turn values are not physically
meaningful for a multi-turn winding pack.

Visualization Links
-------------------

- :math:`i`: Link to 3D visualization plot showing :math:`B_N/|B|` error on plasma surface with initial (pre-optimization) coils
- :math:`f`: Link to 3D visualization plot showing :math:`B_N/|B|` error on plasma surface with final (optimized) coils
- **PP**: Link to Poincaré plot showing fieldline trajectories
- **BP**: Link to Boozer surface plot showing flux surfaces
- **QS**: Link to quasisymmetry error profile plot
- **iota**: Link to rotational transform (iota) profile plot
- **FPT**: Link to Fast Particle Tracing (SIMPLE) loss fraction plot
