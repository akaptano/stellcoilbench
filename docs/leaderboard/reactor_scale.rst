Reactor-Scale Leaderboard
=========================

.. role:: red
.. role:: orange

.. raw:: html

   <style>
   .red { color: #dc3545; font-weight: bold; }
   .orange { color: #e67e22; font-weight: bold; }
   </style>

All values are scaled to the **ARIES-CS reference** (major radius :math:`R_0 = 7.5` m, on-axis field :math:`B_0 = 5.7` T).

Entries are ranked by **composite score** (higher = better engineering margin). See :doc:`metric_definitions` for constraint bounds and the scoring formula.

How constraints are applied
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Hard constraints** make a design *infeasible*.  Any hard-constraint violation sets the composite score to **0** and marks the entry **FAIL**.  Hard constraints test topological validity (coils must encircle the plasma, coils must not interlink) and engineering limits on the winding-pack turns.

**Soft constraints** encode engineering preferences.  Each soft constraint contributes an exponential margin factor to the composite score (see :doc:`metric_definitions`).  A soft-constraint violation lowers the score below 1 but does **not** cause FAIL or exclusion.  Violated soft-constraint cells are highlighted in :orange:`orange`; hard-constraint violations appear in :red:`red`.

Engineering Constraints
-----------------------

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Constraint
     - Bound
     - Type
   * - Coils linked to plasma surface
     - = True
     - hard
   * - Coil-coil linking number (\|LN\| ≈ 0)
     - ≤ 0.5 (dimensionless)
     - hard
   * - avg ⟨B·n⟩/⟨B⟩
     - ≤ 0.01 (dimensionless)
     - soft
   * - Minimum coil-surface distance
     - ≥ 1.3 m
     - soft
   * - Minimum coil-coil distance
     - ≥ 0.7 m
     - soft
   * - Total coil length
     - ≤ 220.0 m
     - soft
   * - Max curvature κ
     - ≤ 1.0 m⁻¹
     - soft
   * - Max √MSC (RMS curvature)
     - ≤ 1.0 m⁻¹
     - soft
   * - Arclength variation √Var
     - ≤ 1.0 m
     - soft
   * - Max turns per coil (N_turns ≤ 500)
     - ≤ 500 (turns)
     - hard
   * - Finite-build coil-coil clearance (d_cc > w_WP)
     - ≥ 0.0 m
     - hard


HSX
---

.. list-table:: HSX — Reactor Scale
   :header-rows: 1
   :widths: auto

   * - :math:`\text{Score}`
     - :math:`\text{Status}`
     - :math:`N`
     - :math:`n`
     - :math:`\bar{B}_n`
     - :math:`d_{cs}\ [\text{m}]`
     - :math:`d_{cc}\ [\text{m}]`
     - :math:`L\ [\text{m}]`
     - :math:`L_\text{SC}\ [\text{km}]`
     - :math:`\kappa_\text{max}\ [\text{m}^{-1}]`
     - :math:`\bar{\kappa}\ [\text{m}^{-1}]`
     - :math:`MSC\ [\text{m}^{-2}]`
     - :math:`F_\text{turn}\ [\text{MN/m}]`
     - :math:`\tau_\text{turn}\ [\text{MN}]`
     - :math:`w_\text{WP}\ [\text{m}]`
     - :math:`\text{LN}`
     - :math:`N_{\text{turns},i}`
     - :math:`\text{User}`
     - :math:`\text{i}`
     - :math:`\text{f}`
     - :math:`\text{PP}`
   * - 1.163
     - pass
     - 5
     - 4
     - :orange:`1.51e-02`
     - :orange:`9.34e-01`
     - :orange:`5.74e-01`
     - 92.87
     - 12.39
     - 7.51e-01
     - 4.40e-01
     - 2.17e-01
     - 3.73e-01
     - 7.06e-01
     - 3.12e-01
     - 0
     - 57, 71, 184, 20, 244
     - akaptano
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/HSX_QHS_mn1824_ns101/akaptano/basic_HSX/02-08-2026_12-50/order_4/bn_error_3d_plot_initial.pdf>`__
     - `4 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/HSX_QHS_mn1824_ns101/akaptano/basic_HSX/02-08-2026_12-50/order_4/bn_error_3d_plot.pdf>`__ `8 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/HSX_QHS_mn1824_ns101/akaptano/basic_HSX/02-08-2026_12-50/order_8/bn_error_3d_plot.pdf>`__
     - —


Landreman-Paul QA
-----------------

.. list-table:: Landreman-Paul QA — Reactor Scale
   :header-rows: 1
   :widths: auto

   * - :math:`\text{Score}`
     - :math:`\text{Status}`
     - :math:`N`
     - :math:`n`
     - :math:`\bar{B}_n`
     - :math:`d_{cs}\ [\text{m}]`
     - :math:`d_{cc}\ [\text{m}]`
     - :math:`L\ [\text{m}]`
     - :math:`L_\text{SC}\ [\text{km}]`
     - :math:`\kappa_\text{max}\ [\text{m}^{-1}]`
     - :math:`\bar{\kappa}\ [\text{m}^{-1}]`
     - :math:`MSC\ [\text{m}^{-2}]`
     - :math:`F_\text{turn}\ [\text{MN/m}]`
     - :math:`\tau_\text{turn}\ [\text{MN}]`
     - :math:`w_\text{WP}\ [\text{m}]`
     - :math:`\text{LN}`
     - :math:`N_{\text{turns},i}`
     - :math:`\text{User}`
     - :math:`\text{i}`
     - :math:`\text{f}`
     - :math:`\text{PP}`
   * - 1.836
     - pass
     - 3
     - 8
     - 3.79e-04
     - 2.54
     - :orange:`6.00e-01`
     - 150.0
     - 53.86
     - 4.38e-01
     - 2.23e-01
     - 7.88e-02
     - 4.19e-01
     - 2.27
     - 4.26e-01
     - 0
     - 295, 453, 322
     - akaptano
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/case/02-09-2026_21-39/bn_error_3d_plot_initial.pdf>`__
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/case/02-09-2026_21-39/bn_error_3d_plot.pdf>`__
     - —
   * - 1.836
     - pass
     - 3
     - 8
     - 3.79e-04
     - 2.54
     - :orange:`6.00e-01`
     - 150.0
     - 53.86
     - 4.38e-01
     - 2.23e-01
     - 7.88e-02
     - 4.19e-01
     - 2.27
     - 4.26e-01
     - 0
     - 295, 453, 322
     - akaptano
     - `2 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/case/02-09-2026_20-25/bn_error_3d_plot_initial.pdf>`__
     - `2 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/case/02-09-2026_20-25/bn_error_3d_plot.pdf>`__
     - —
   * - 1.740
     - pass
     - 4
     - 4
     - 7.77e-04
     - 2.13
     - :orange:`5.98e-01`
     - 150.0
     - 43.99
     - 3.75e-01
     - 2.33e-01
     - 8.36e-02
     - 4.96e-01
     - 1.94
     - 4.12e-01
     - 0
     - 425, 124, 330, 224
     - auto
     - `3 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/auto/2026-02-09_201853_47056/bn_error_3d_plot_initial.pdf>`__
     - `3 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/auto/2026-02-09_201853_47056/bn_error_3d_plot.pdf>`__
     - —
   * - 1.728
     - pass
     - 3
     - 12
     - 5.00e-04
     - 2.49
     - :orange:`6.00e-01`
     - 150.0
     - 53.78
     - 7.54e-01
     - 2.62e-01
     - 1.08e-01
     - 4.73e-01
     - 3.47
     - 4.14e-01
     - 0
     - 335, 307, 428
     - akaptano
     - `4 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/case/02-09-2026_21-44/bn_error_3d_plot_initial.pdf>`__
     - `4 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/case/02-09-2026_21-44/bn_error_3d_plot.pdf>`__
     - —
   * - 1.718
     - pass
     - 4
     - 16
     - 3.49e-04
     - 2.32
     - :orange:`5.97e-01`
     - 150.0
     - 41.37
     - 7.33e-01
     - 2.77e-01
     - 1.01e-01
     - 4.98e-01
     - 1.65
     - 3.42e-01
     - 0
     - 256, 266, 293, 288
     - akaptano
     - `5 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/expert_LandremanPaulQA/02-08-2026_12-58/order_4/bn_error_3d_plot_initial.pdf>`__
     - `4 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/expert_LandremanPaulQA/02-08-2026_12-58/order_4/bn_error_3d_plot.pdf>`__ `8 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/expert_LandremanPaulQA/02-08-2026_12-58/order_8/bn_error_3d_plot.pdf>`__ `12 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/expert_LandremanPaulQA/02-08-2026_12-58/order_12/bn_error_3d_plot.pdf>`__ `16 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/expert_LandremanPaulQA/02-08-2026_12-58/order_16/bn_error_3d_plot.pdf>`__
     - —
   * - 1.711
     - pass
     - 4
     - 8
     - 3.51e-04
     - 2.36
     - :orange:`6.00e-01`
     - 150.0
     - 40.94
     - 8.12e-01
     - 2.80e-01
     - 1.12e-01
     - 4.99e-01
     - 1.74
     - 3.39e-01
     - 0
     - 260, 281, 264, 287
     - akaptano
     - `6 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/advanced_LandremanPaulQA/02-08-2026_12-50/order_4/bn_error_3d_plot_initial.pdf>`__
     - `4 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/advanced_LandremanPaulQA/02-08-2026_12-50/order_4/bn_error_3d_plot.pdf>`__ `8 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/advanced_LandremanPaulQA/02-08-2026_12-50/order_8/bn_error_3d_plot.pdf>`__ `16 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/advanced_LandremanPaulQA/02-08-2026_12-50/order_16/bn_error_3d_plot.pdf>`__
     - —
   * - 1.658
     - pass
     - 4
     - 8
     - 8.14e-04
     - 2.03
     - :orange:`5.99e-01`
     - 150.0
     - 42.10
     - 5.99e-01
     - 2.81e-01
     - 1.11e-01
     - 4.99e-01
     - 2.56
     - 3.84e-01
     - 0
     - 145, 296, 368, 287
     - auto
     - `7 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/auto/2026-02-09_201853_99682/bn_error_3d_plot_initial.pdf>`__
     - `7 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/auto/2026-02-09_201853_99682/bn_error_3d_plot.pdf>`__
     - —
   * - 1.641
     - pass
     - 4
     - 12
     - 4.64e-04
     - 2.18
     - :orange:`6.00e-01`
     - 150.0
     - 41.99
     - 8.63e-01
     - 2.84e-01
     - 1.36e-01
     - 4.99e-01
     - 2.81
     - 3.64e-01
     - 0
     - 237, 251, 331, 298
     - auto
     - `8 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/auto/2026-02-09_201853_60452/bn_error_3d_plot_initial.pdf>`__
     - `8 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/auto/2026-02-09_201853_60452/bn_error_3d_plot.pdf>`__
     - —
   * - 1.636
     - pass
     - 3
     - 16
     - 4.89e-04
     - 2.36
     - :orange:`6.00e-01`
     - 150.1
     - 55.03
     - 9.32e-01
     - 3.25e-01
     - 1.78e-01
     - 4.99e-01
     - 3.61
     - 4.21e-01
     - 0
     - 329, 316, 444
     - akaptano
     - `9 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/case/02-09-2026_20-27/bn_error_3d_plot_initial.pdf>`__
     - `9 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/case/02-09-2026_20-27/bn_error_3d_plot.pdf>`__
     - —
   * - 1.624
     - pass
     - 4
     - 6
     - 8.24e-04
     - 1.87
     - :orange:`5.98e-01`
     - 150.0
     - 42.84
     - 5.86e-01
     - 2.54e-01
     - 1.31e-01
     - 4.99e-01
     - 2.33
     - 3.93e-01
     - 0
     - 239, 387, 91, 377
     - auto
     - `10 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/auto/2026-02-09_201853_29256/bn_error_3d_plot_initial.pdf>`__
     - `10 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/auto/2026-02-09_201853_29256/bn_error_3d_plot.pdf>`__
     - —
   * - 1.550
     - pass
     - 6
     - 4
     - 1.12e-03
     - 1.35
     - :orange:`6.41e-01`
     - 150.0
     - 27.86
     - 5.16e-01
     - 3.14e-01
     - 1.52e-01
     - 2.79e-01
     - 5.71e-01
     - 3.07e-01
     - 0
     - 235, 229, 197, 162, 131, 118
     - auto
     - `11 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/auto/2026-02-09_201853_41285/bn_error_3d_plot_initial.pdf>`__
     - `11 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/auto/2026-02-09_201853_41285/bn_error_3d_plot.pdf>`__
     - —
   * - 1.445
     - pass
     - 6
     - 16
     - 1.26e-03
     - 1.60
     - :orange:`5.96e-01`
     - 150.0
     - 26.81
     - :orange:`1.14`
     - 3.46e-01
     - 1.78e-01
     - 2.87e-01
     - 6.36e-01
     - 2.91e-01
     - 0
     - 164, 166, 182, 167, 182, 211
     - akaptano
     - `12 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/case/02-09-2026_20-19/bn_error_3d_plot_initial.pdf>`__
     - `12 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/case/02-09-2026_20-19/bn_error_3d_plot.pdf>`__
     - —
   * - 1.432
     - pass
     - 7
     - 16
     - 1.99e-03
     - 1.34
     - :orange:`5.96e-01`
     - 150.0
     - 22.96
     - 8.97e-01
     - 3.69e-01
     - 1.78e-01
     - 3.51e-01
     - 4.04e-01
     - 2.58e-01
     - 0
     - 155, 156, 140, 150, 160, 144, 166
     - auto
     - `13 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/auto/2026-02-09_201853_27728/bn_error_3d_plot_initial.pdf>`__
     - `13 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/auto/2026-02-09_201853_27728/bn_error_3d_plot.pdf>`__
     - —
   * - 0.000
     - :red:`FAIL`
     - 2
     - 6
     - 8.79e-04
     - 2.85
     - :orange:`5.99e-01`
     - 150.0
     - 92.84
     - 2.58e-01
     - 1.59e-01
     - 2.97e-02
     - 5.00e-01
     - 3.72
     - 5.49e-01
     - 0
     - :red:`753, 438`
     - auto
     - `14 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/auto/2026-02-09_201853_65617/bn_error_3d_plot_initial.pdf>`__
     - `14 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/auto/2026-02-09_201853_65617/bn_error_3d_plot.pdf>`__
     - —
   * - 0.000
     - :red:`FAIL`
     - 2
     - 4
     - 1.29e-03
     - 2.85
     - :orange:`5.98e-01`
     - 150.0
     - 101.9
     - 2.51e-01
     - 1.32e-01
     - 2.85e-02
     - 5.00e-01
     - 3.59
     - 6.01e-01
     - 0
     - :red:`904, 364`
     - auto
     - `15 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/auto/2026-02-09_201853_86167/bn_error_3d_plot_initial.pdf>`__
     - `15 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/auto/2026-02-09_201853_86167/bn_error_3d_plot.pdf>`__
     - —
   * - 0.000
     - :red:`FAIL`
     - 2
     - 8
     - 9.25e-04
     - 2.81
     - :orange:`5.96e-01`
     - 146.9
     - 89.90
     - 3.90e-01
     - 2.08e-01
     - 6.59e-02
     - 5.00e-01
     - 3.47
     - 5.76e-01
     - 0
     - :red:`373, 829`
     - akaptano
     - `16 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/case/02-09-2026_21-41/bn_error_3d_plot_initial.pdf>`__
     - `16 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/case/02-09-2026_21-41/bn_error_3d_plot.pdf>`__
     - —


Landreman-Paul QH
-----------------

.. list-table:: Landreman-Paul QH — Reactor Scale
   :header-rows: 1
   :widths: auto

   * - :math:`\text{Score}`
     - :math:`\text{Status}`
     - :math:`N`
     - :math:`n`
     - :math:`\bar{B}_n`
     - :math:`d_{cs}\ [\text{m}]`
     - :math:`d_{cc}\ [\text{m}]`
     - :math:`L\ [\text{m}]`
     - :math:`L_\text{SC}\ [\text{km}]`
     - :math:`\kappa_\text{max}\ [\text{m}^{-1}]`
     - :math:`\bar{\kappa}\ [\text{m}^{-1}]`
     - :math:`MSC\ [\text{m}^{-2}]`
     - :math:`F_\text{turn}\ [\text{MN/m}]`
     - :math:`\tau_\text{turn}\ [\text{MN}]`
     - :math:`w_\text{WP}\ [\text{m}]`
     - :math:`\text{LN}`
     - :math:`N_{\text{turns},i}`
     - :math:`\text{User}`
     - :math:`\text{i}`
     - :math:`\text{f}`
     - :math:`\text{PP}`
   * - 1.159
     - pass
     - 5
     - 4
     - 1.01e-03
     - :orange:`9.74e-01`
     - :orange:`5.88e-01`
     - 98.80
     - 10.79
     - :orange:`1.24`
     - 4.22e-01
     - 2.90e-01
     - 2.58e-01
     - 5.87e-01
     - 2.35e-01
     - 0
     - 138, 97, 127, 71, 104
     - akaptano
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QH_reactorScale_lowres/akaptano/basic_LandremanPaulQH/02-08-2026_12-50/bn_error_3d_plot_initial.pdf>`__
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QH_reactorScale_lowres/akaptano/basic_LandremanPaulQH/02-08-2026_12-50/bn_error_3d_plot.pdf>`__
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QH_reactorScale_lowres/akaptano/basic_LandremanPaulQH/02-08-2026_12-50/poincare_plot.png>`__


W7-X
----

.. list-table:: W7-X — Reactor Scale
   :header-rows: 1
   :widths: auto

   * - :math:`\text{Score}`
     - :math:`\text{Status}`
     - :math:`N`
     - :math:`n`
     - :math:`\bar{B}_n`
     - :math:`d_{cs}\ [\text{m}]`
     - :math:`d_{cc}\ [\text{m}]`
     - :math:`L\ [\text{m}]`
     - :math:`L_\text{SC}\ [\text{km}]`
     - :math:`\kappa_\text{max}\ [\text{m}^{-1}]`
     - :math:`\bar{\kappa}\ [\text{m}^{-1}]`
     - :math:`MSC\ [\text{m}^{-2}]`
     - :math:`F_\text{turn}\ [\text{MN/m}]`
     - :math:`\tau_\text{turn}\ [\text{MN}]`
     - :math:`w_\text{WP}\ [\text{m}]`
     - :math:`\text{LN}`
     - :math:`N_{\text{turns},i}`
     - :math:`\text{User}`
     - :math:`\text{i}`
     - :math:`\text{f}`
     - :math:`\text{PP}`
   * - 1.088
     - pass
     - 4
     - 4
     - 3.93e-03
     - :orange:`4.51e-01`
     - :orange:`2.99e-01`
     - 61.41
     - 6.72
     - :orange:`1.36`
     - 8.85e-01
     - 9.88e-01
     - 3.54e-01
     - 5.24e-01
     - 2.52e-01
     - 0
     - 124, 159, 85, 61
     - akaptano
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/W7-X_without_coil_ripple_beta0p05_d23p4_tm/akaptano/expert_W7X/02-08-2026_12-58/order_4/bn_error_3d_plot_initial.pdf>`__
     - `4 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/W7-X_without_coil_ripple_beta0p05_d23p4_tm/akaptano/expert_W7X/02-08-2026_12-58/order_4/bn_error_3d_plot.pdf>`__ `8 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/W7-X_without_coil_ripple_beta0p05_d23p4_tm/akaptano/expert_W7X/02-08-2026_12-58/order_8/bn_error_3d_plot.pdf>`__ `16 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/W7-X_without_coil_ripple_beta0p05_d23p4_tm/akaptano/expert_W7X/02-08-2026_12-58/order_16/bn_error_3d_plot.pdf>`__
     - —
   * - 1.027
     - pass
     - 4
     - 4
     - 3.85e-03
     - :orange:`4.73e-01`
     - :orange:`3.24e-01`
     - 61.42
     - 6.70
     - :orange:`1.33`
     - 8.15e-01
     - 9.51e-01
     - 3.21e-01
     - 4.08e-01
     - 2.58e-01
     - 0
     - 94, 166, 105, 65
     - akaptano
     - `2 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/W7-X_without_coil_ripple_beta0p05_d23p4_tm/akaptano/basic_W7X/02-08-2026_12-57/order_4/bn_error_3d_plot_initial.pdf>`__
     - `4 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/W7-X_without_coil_ripple_beta0p05_d23p4_tm/akaptano/basic_W7X/02-08-2026_12-57/order_4/bn_error_3d_plot.pdf>`__ `8 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/W7-X_without_coil_ripple_beta0p05_d23p4_tm/akaptano/basic_W7X/02-08-2026_12-57/order_8/bn_error_3d_plot.pdf>`__
     - —


0.5 Tesla NCSX Design
---------------------

.. list-table:: 0.5 Tesla NCSX Design — Reactor Scale
   :header-rows: 1
   :widths: auto

   * - :math:`\text{Score}`
     - :math:`\text{Status}`
     - :math:`N`
     - :math:`n`
     - :math:`\bar{B}_n`
     - :math:`d_{cs}\ [\text{m}]`
     - :math:`d_{cc}\ [\text{m}]`
     - :math:`L\ [\text{m}]`
     - :math:`L_\text{SC}\ [\text{km}]`
     - :math:`\kappa_\text{max}\ [\text{m}^{-1}]`
     - :math:`\bar{\kappa}\ [\text{m}^{-1}]`
     - :math:`MSC\ [\text{m}^{-2}]`
     - :math:`F_\text{turn}\ [\text{MN/m}]`
     - :math:`\tau_\text{turn}\ [\text{MN}]`
     - :math:`w_\text{WP}\ [\text{m}]`
     - :math:`\text{LN}`
     - :math:`N_{\text{turns},i}`
     - :math:`\text{User}`
     - :math:`\text{i}`
     - :math:`\text{f}`
     - :math:`\text{PP}`
   * - 1.292
     - pass
     - 4
     - 4
     - 5.05e-03
     - :orange:`9.74e-01`
     - :orange:`5.98e-01`
     - 159.6
     - 30.03
     - 7.45e-01
     - 3.99e-01
     - 2.51e-01
     - 4.44e-01
     - 1.15
     - 3.62e-01
     - 0
     - 190, 328, 102, 94
     - akaptano
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/c09r00_B_axis_half_tesla_NCSX/akaptano/basic_NCSX/02-08-2026_12-50/order_4/bn_error_3d_plot_initial.pdf>`__
     - `4 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/c09r00_B_axis_half_tesla_NCSX/akaptano/basic_NCSX/02-08-2026_12-50/order_4/bn_error_3d_plot.pdf>`__ `8 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/c09r00_B_axis_half_tesla_NCSX/akaptano/basic_NCSX/02-08-2026_12-50/order_8/bn_error_3d_plot.pdf>`__
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/c09r00_B_axis_half_tesla_NCSX/akaptano/basic_NCSX/02-08-2026_12-50/poincare_plot.png>`__


CFQS
----

.. list-table:: CFQS — Reactor Scale
   :header-rows: 1
   :widths: auto

   * - :math:`\text{Score}`
     - :math:`\text{Status}`
     - :math:`N`
     - :math:`n`
     - :math:`\bar{B}_n`
     - :math:`d_{cs}\ [\text{m}]`
     - :math:`d_{cc}\ [\text{m}]`
     - :math:`L\ [\text{m}]`
     - :math:`L_\text{SC}\ [\text{km}]`
     - :math:`\kappa_\text{max}\ [\text{m}^{-1}]`
     - :math:`\bar{\kappa}\ [\text{m}^{-1}]`
     - :math:`MSC\ [\text{m}^{-2}]`
     - :math:`F_\text{turn}\ [\text{MN/m}]`
     - :math:`\tau_\text{turn}\ [\text{MN}]`
     - :math:`w_\text{WP}\ [\text{m}]`
     - :math:`\text{LN}`
     - :math:`N_{\text{turns},i}`
     - :math:`\text{User}`
     - :math:`\text{i}`
     - :math:`\text{f}`
     - :math:`\text{PP}`
   * - 1.461
     - pass
     - 4
     - 8
     - 3.38e-03
     - 1.47
     - :orange:`5.97e-01`
     - 150.0
     - 40.25
     - 6.82e-01
     - 2.78e-01
     - 1.47e-01
     - 4.33e-01
     - 2.00
     - 3.45e-01
     - 0
     - 256, 278, 297, 239
     - akaptano
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/cfqs_2b40/akaptano/basic_CFQS/02-08-2026_12-50/bn_error_3d_plot_initial.pdf>`__
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/cfqs_2b40/akaptano/basic_CFQS/02-08-2026_12-50/bn_error_3d_plot.pdf>`__
     - —


Circular Tokamak
----------------

.. list-table:: Circular Tokamak — Reactor Scale
   :header-rows: 1
   :widths: auto

   * - :math:`\text{Score}`
     - :math:`\text{Status}`
     - :math:`N`
     - :math:`n`
     - :math:`\bar{B}_n`
     - :math:`d_{cs}\ [\text{m}]`
     - :math:`d_{cc}\ [\text{m}]`
     - :math:`L\ [\text{m}]`
     - :math:`L_\text{SC}\ [\text{km}]`
     - :math:`\kappa_\text{max}\ [\text{m}^{-1}]`
     - :math:`\bar{\kappa}\ [\text{m}^{-1}]`
     - :math:`MSC\ [\text{m}^{-2}]`
     - :math:`F_\text{turn}\ [\text{MN/m}]`
     - :math:`\tau_\text{turn}\ [\text{MN}]`
     - :math:`w_\text{WP}\ [\text{m}]`
     - :math:`\text{LN}`
     - :math:`N_{\text{turns},i}`
     - :math:`\text{User}`
     - :math:`\text{i}`
     - :math:`\text{f}`
     - :math:`\text{PP}`
   * - 2.006
     - pass
     - 6
     - 4
     - 4.12e-03
     - 1.96
     - 1.57
     - :orange:`225.0`
     - 80.26
     - 1.75e-01
     - 1.68e-01
     - 2.84e-02
     - 3.26e-01
     - 3.91e-02
     - 3.79e-01
     - 0
     - 359, 352, 359, 359, 352, 359
     - akaptano
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/circular_tokamak/akaptano/basic_tokamak/02-08-2026_12-56/bn_error_3d_plot_initial.pdf>`__
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/circular_tokamak/akaptano/basic_tokamak/02-08-2026_12-56/bn_error_3d_plot.pdf>`__
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/circular_tokamak/akaptano/basic_tokamak/02-08-2026_12-56/poincare_plot.png>`__


MUSE
----

.. list-table:: MUSE — Reactor Scale
   :header-rows: 1
   :widths: auto

   * - :math:`\text{Score}`
     - :math:`\text{Status}`
     - :math:`N`
     - :math:`n`
     - :math:`\bar{B}_n`
     - :math:`d_{cs}\ [\text{m}]`
     - :math:`d_{cc}\ [\text{m}]`
     - :math:`L\ [\text{m}]`
     - :math:`L_\text{SC}\ [\text{km}]`
     - :math:`\kappa_\text{max}\ [\text{m}^{-1}]`
     - :math:`\bar{\kappa}\ [\text{m}^{-1}]`
     - :math:`MSC\ [\text{m}^{-2}]`
     - :math:`F_\text{turn}\ [\text{MN/m}]`
     - :math:`\tau_\text{turn}\ [\text{MN}]`
     - :math:`w_\text{WP}\ [\text{m}]`
     - :math:`\text{LN}`
     - :math:`N_{\text{turns},i}`
     - :math:`\text{User}`
     - :math:`\text{i}`
     - :math:`\text{f}`
     - :math:`\text{PP}`
   * - 1.632
     - pass
     - 4
     - 8
     - :orange:`1.05e-02`
     - 2.60
     - :orange:`6.01e-01`
     - 150.0
     - 42.21
     - 4.10e-01
     - 2.19e-01
     - 5.42e-02
     - 5.00e-01
     - 2.48
     - 4.00e-01
     - 0
     - 249, 232, 401, 228
     - akaptano
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/muse/akaptano/basic_MUSE/02-08-2026_12-50/bn_error_3d_plot_initial.pdf>`__
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/muse/akaptano/basic_MUSE/02-08-2026_12-50/bn_error_3d_plot.pdf>`__
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/muse/akaptano/basic_MUSE/02-08-2026_12-50/poincare_plot.png>`__


Rotating Ellipse
----------------

.. list-table:: Rotating Ellipse — Reactor Scale
   :header-rows: 1
   :widths: auto

   * - :math:`\text{Score}`
     - :math:`\text{Status}`
     - :math:`N`
     - :math:`n`
     - :math:`\bar{B}_n`
     - :math:`d_{cs}\ [\text{m}]`
     - :math:`d_{cc}\ [\text{m}]`
     - :math:`L\ [\text{m}]`
     - :math:`L_\text{SC}\ [\text{km}]`
     - :math:`\kappa_\text{max}\ [\text{m}^{-1}]`
     - :math:`\bar{\kappa}\ [\text{m}^{-1}]`
     - :math:`MSC\ [\text{m}^{-2}]`
     - :math:`F_\text{turn}\ [\text{MN/m}]`
     - :math:`\tau_\text{turn}\ [\text{MN}]`
     - :math:`w_\text{WP}\ [\text{m}]`
     - :math:`\text{LN}`
     - :math:`N_{\text{turns},i}`
     - :math:`\text{User}`
     - :math:`\text{i}`
     - :math:`\text{f}`
     - :math:`\text{PP}`
   * - 0.000
     - :red:`FAIL`
     - 4
     - 8
     - 9.56e-04
     - :orange:`1.03`
     - :orange:`5.99e-01`
     - 150.0
     - 75.70
     - 7.06e-01
     - 2.65e-01
     - 1.18e-01
     - 4.99e-01
     - 2.56
     - 6.14e-01
     - 0
     - :red:`673, 944, 41, 203`
     - akaptano
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/rotating_ellipse/akaptano/basic_rotating_ellipse/02-08-2026_12-50/bn_error_3d_plot_initial.pdf>`__
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/rotating_ellipse/akaptano/basic_rotating_ellipse/02-08-2026_12-50/bn_error_3d_plot.pdf>`__
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/rotating_ellipse/akaptano/basic_rotating_ellipse/02-08-2026_12-50/poincare_plot.png>`__


Schuett-Henneberg QA
--------------------

.. list-table:: Schuett-Henneberg QA — Reactor Scale
   :header-rows: 1
   :widths: auto

   * - :math:`\text{Score}`
     - :math:`\text{Status}`
     - :math:`N`
     - :math:`n`
     - :math:`\bar{B}_n`
     - :math:`d_{cs}\ [\text{m}]`
     - :math:`d_{cc}\ [\text{m}]`
     - :math:`L\ [\text{m}]`
     - :math:`L_\text{SC}\ [\text{km}]`
     - :math:`\kappa_\text{max}\ [\text{m}^{-1}]`
     - :math:`\bar{\kappa}\ [\text{m}^{-1}]`
     - :math:`MSC\ [\text{m}^{-2}]`
     - :math:`F_\text{turn}\ [\text{MN/m}]`
     - :math:`\tau_\text{turn}\ [\text{MN}]`
     - :math:`w_\text{WP}\ [\text{m}]`
     - :math:`\text{LN}`
     - :math:`N_{\text{turns},i}`
     - :math:`\text{User}`
     - :math:`\text{i}`
     - :math:`\text{f}`
     - :math:`\text{PP}`
   * - 1.472
     - pass
     - 4
     - 8
     - 5.14e-04
     - 2.03
     - :orange:`5.99e-01`
     - :orange:`227.5`
     - 60.94
     - 5.45e-01
     - 2.25e-01
     - 8.60e-02
     - 4.36e-01
     - 2.32
     - 3.38e-01
     - 0
     - 285, 262, 253, 271
     - akaptano
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/wout_schuetthenneberg_nfp2/akaptano/basic_SchuettHennebergQA_nfp2/02-08-2026_12-55/bn_error_3d_plot_initial.pdf>`__
     - —
     - —


.. note::
   Last updated: run ``stellcoilbench update-db`` to refresh locally.
