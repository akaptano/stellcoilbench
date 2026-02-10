Surface-Specific Leaderboards
===============================

Each plasma surface presents unique challenges for coil optimization. The following
tables show detailed results for each surface, allowing for direct comparison
of methods on specific configurations.

Visualization Links
--------------------

The leaderboard tables include visualization links in the following columns:

- :math:`i`: Link to 3D visualization plot showing :math:`B_N/|B|` error on plasma surface with initial (pre-optimization) coils
- :math:`f`: Link to 3D visualization plot showing :math:`B_N/|B|` error on plasma surface with final (optimized) coils
- **PP**: Link to Poincaré plot showing fieldline trajectories
- **BP**: Link to Boozer surface plot showing flux surfaces
- **QS**: Link to quasisymmetry error profile plot
- **iota**: Link to rotational transform (iota) profile plot
- **FPT**: Link to Fast Particle Tracing (SIMPLE) loss fraction plot

.. _hsx-qhs-mn1824-ns101:

HSX
^^^

**Surface file:** ``HSX_QHS_mn1824_ns101``

This surface has 1 submission(s).
Typical configuration: 4 Fourier order, 5 base coils.

.. list-table:: HSX Leaderboard
   :header-rows: 1
   :widths: auto

   * - :math:`\text{Score}`
     - :math:`N`
     - :math:`n`
     - :math:`\text{FC}`
     - :math:`f_{B}`
     - :math:`\bar{B}_n`
     - :math:`\max(B_n)`
     - :math:`L`
     - :math:`\mathrm{Var}(l_i)`
     - :math:`d_{cc}`
     - :math:`d_{cs}`
     - :math:`\bar{\kappa}`
     - :math:`MSC`
     - :math:`\bar{F}`
     - :math:`\bar{\tau}`
     - :math:`F_\text{max}`
     - :math:`\tau_\text{max}`
     - :math:`LN`
     - :math:`t`
     - :math:`\kappa_\text{max}`
     - :math:`\text{Date}`
     - :math:`\text{User}`
     - :math:`\text{i}`
     - :math:`\text{f}`
     - :math:`\text{PP}`
     - :math:`\text{BP}`
     - :math:`\text{QS}`
     - :math:`\text{iota}`
     - :math:`\text{FPT}`
   * - 1.163
     - 5
     - 4
     - 4,8
     - 3.7e-03
     - 1.5e-02
     - 8.7e-02
     - 1.5e+01
     - 3.0e-03
     - 9.3e-02
     - 1.5e-01
     - 2.7e+00
     - 8.3e+00
     - 5.5e+05
     - 1.7e+05
     - 1.0e+06
     - 4.9e+05
     - 0
     - 5.6e+02
     - 4.6e+00
     - 08/02/26
     - akaptano
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/HSX_QHS_mn1824_ns101/akaptano/basic_HSX/02-08-2026_12-50/order_4/bn_error_3d_plot_initial.pdf>`__
     - `4 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/HSX_QHS_mn1824_ns101/akaptano/basic_HSX/02-08-2026_12-50/order_4/bn_error_3d_plot.pdf>`__ `8 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/HSX_QHS_mn1824_ns101/akaptano/basic_HSX/02-08-2026_12-50/order_8/bn_error_3d_plot.pdf>`__
     - —
     - —
     - —
     - —
     - —


.. _landremanpaul2021-qa:

Landreman-Paul QA
^^^^^^^^^^^^^^^^^

**Surface file:** ``LandremanPaul2021_QA``

This surface has 11 submission(s).
Typical configuration: 8 Fourier order, 3 base coils.

.. list-table:: Landreman-Paul QA Leaderboard
   :header-rows: 1
   :widths: auto

   * - :math:`\text{Score}`
     - :math:`N`
     - :math:`n`
     - :math:`\text{FC}`
     - :math:`f_{B}`
     - :math:`\bar{B}_n`
     - :math:`\max(B_n)`
     - :math:`L`
     - :math:`\mathrm{Var}(l_i)`
     - :math:`d_{cc}`
     - :math:`d_{cs}`
     - :math:`\bar{\kappa}`
     - :math:`MSC`
     - :math:`\bar{F}`
     - :math:`\bar{\tau}`
     - :math:`F_\text{max}`
     - :math:`\tau_\text{max}`
     - :math:`LN`
     - :math:`t`
     - :math:`\kappa_\text{max}`
     - :math:`\text{iterations\ used}`
     - :math:`\text{walltime\ sec}`
     - :math:`\text{Date}`
     - :math:`\text{User}`
     - :math:`\text{i}`
     - :math:`\text{f}`
     - :math:`\text{PP}`
     - :math:`\text{BP}`
     - :math:`\text{QS}`
     - :math:`\text{iota}`
     - :math:`\text{FPT}`
   * - 1.836
     - 3
     - 8
     - —
     - 7.7e-07
     - 3.8e-04
     - 1.6e-03
     - 2.0e+01
     - 2.5e-04
     - 8.0e-02
     - 3.4e-01
     - 1.7e+00
     - 4.4e+00
     - 6.1e+05
     - 3.6e+05
     - 7.7e+05
     - 4.7e+05
     - 0
     - 2.4e+02
     - 3.3e+00
     - 5.0e+02
     - 2.4e+02
     - 09/02/26
     - akaptano
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/case/02-09-2026_20-25/bn_error_3d_plot_initial.pdf>`__
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/case/02-09-2026_20-25/bn_error_3d_plot.pdf>`__
     - —
     - —
     - —
     - —
     - —
   * - 1.718
     - 4
     - 16
     - 4,8,12,16
     - 6.4e-07
     - 3.5e-04
     - 1.4e-03
     - 2.0e+01
     - 1.1e-04
     - 8.0e-02
     - 3.1e-01
     - 2.1e+00
     - 5.7e+00
     - 4.3e+05
     - 1.7e+05
     - 5.4e+05
     - 2.4e+05
     - 0
     - 1.5e+03
     - 5.5e+00
     - —
     - —
     - 08/02/26
     - akaptano
     - `2 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/expert_LandremanPaulQA/02-08-2026_12-58/order_4/bn_error_3d_plot_initial.pdf>`__
     - `4 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/expert_LandremanPaulQA/02-08-2026_12-58/order_4/bn_error_3d_plot.pdf>`__ `8 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/expert_LandremanPaulQA/02-08-2026_12-58/order_8/bn_error_3d_plot.pdf>`__ `12 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/expert_LandremanPaulQA/02-08-2026_12-58/order_12/bn_error_3d_plot.pdf>`__ `16 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/expert_LandremanPaulQA/02-08-2026_12-58/order_16/bn_error_3d_plot.pdf>`__
     - —
     - —
     - —
     - —
     - —
   * - 1.711
     - 4
     - 8
     - 4,8,16
     - 7.0e-07
     - 3.5e-04
     - 1.4e-03
     - 2.0e+01
     - 4.2e-05
     - 8.0e-02
     - 3.1e-01
     - 2.1e+00
     - 6.3e+00
     - 4.4e+05
     - 1.7e+05
     - 5.8e+05
     - 2.7e+05
     - 0
     - 1.8e+02
     - 6.1e+00
     - —
     - —
     - 08/02/26
     - akaptano
     - `3 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/advanced_LandremanPaulQA/02-08-2026_12-50/order_4/bn_error_3d_plot_initial.pdf>`__
     - `4 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/advanced_LandremanPaulQA/02-08-2026_12-50/order_4/bn_error_3d_plot.pdf>`__ `8 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/advanced_LandremanPaulQA/02-08-2026_12-50/order_8/bn_error_3d_plot.pdf>`__ `16 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/advanced_LandremanPaulQA/02-08-2026_12-50/order_16/bn_error_3d_plot.pdf>`__
     - —
     - —
     - —
     - —
     - —
   * - 1.670
     - 5
     - 6
     - —
     - 3.7e-06
     - 8.6e-04
     - 3.1e-03
     - 2.0e+01
     - 5.0e-04
     - 8.0e-02
     - 2.7e-01
     - 2.0e+00
     - 5.7e+00
     - 2.2e+05
     - 6.2e+04
     - 2.6e+05
     - 7.2e+04
     - 0
     - 2.4e+02
     - 4.3e+00
     - —
     - —
     - 09/02/26
     - auto
     - `4 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/auto/2026-02-09_230708_58477/bn_error_3d_plot_initial.pdf>`__
     - `4 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/auto/2026-02-09_230708_58477/bn_error_3d_plot.pdf>`__
     - —
     - —
     - —
     - —
     - —
   * - 1.636
     - 3
     - 16
     - —
     - 1.4e-06
     - 4.9e-04
     - 2.1e-03
     - 2.0e+01
     - 3.7e-04
     - 8.0e-02
     - 3.1e-01
     - 2.4e+00
     - 1.0e+01
     - 7.0e+05
     - 5.5e+05
     - 9.1e+05
     - 8.8e+05
     - 0
     - 1.1e+02
     - 7.0e+00
     - 5.0e+02
     - 1.1e+02
     - 09/02/26
     - akaptano
     - `5 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/case/02-09-2026_20-27/bn_error_3d_plot_initial.pdf>`__
     - `5 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/case/02-09-2026_20-27/bn_error_3d_plot.pdf>`__
     - —
     - —
     - —
     - —
     - —
   * - 1.624
     - 4
     - 6
     - —
     - 3.4e-06
     - 8.2e-04
     - 3.3e-03
     - 2.0e+01
     - 6.0e-04
     - 8.0e-02
     - 2.5e-01
     - 1.9e+00
     - 7.4e+00
     - 4.8e+05
     - 2.5e+05
     - 7.9e+05
     - 3.8e+05
     - 0
     - 1.8e+02
     - 4.4e+00
     - —
     - —
     - 09/02/26
     - auto
     - `6 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/auto/2026-02-09_230708_41154/bn_error_3d_plot_initial.pdf>`__
     - `6 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/auto/2026-02-09_230708_41154/bn_error_3d_plot.pdf>`__
     - —
     - —
     - —
     - —
     - —
   * - 1.603
     - 7
     - 12
     - —
     - 1.2e-05
     - 1.7e-03
     - 6.0e-03
     - 2.0e+01
     - 8.0e-05
     - 1.4e-01
     - 1.5e-01
     - 2.6e+00
     - 1.0e+01
     - 1.3e+05
     - 3.0e+04
     - 1.7e+05
     - 5.4e+04
     - 0
     - 6.0e+02
     - 5.1e+00
     - —
     - —
     - 09/02/26
     - auto
     - `7 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/auto/2026-02-09_230708_40665/bn_error_3d_plot_initial.pdf>`__
     - `7 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/auto/2026-02-09_230708_40665/bn_error_3d_plot.pdf>`__
     - —
     - —
     - —
     - —
     - —
   * - 1.562
     - 5
     - 8
     - —
     - 2.9e-06
     - 7.7e-04
     - 2.8e-03
     - 2.0e+01
     - 3.4e-04
     - 8.0e-02
     - 2.2e-01
     - 2.1e+00
     - 8.1e+00
     - 2.4e+05
     - 7.9e+04
     - 3.1e+05
     - 1.0e+05
     - 0
     - 1.9e+02
     - 5.5e+00
     - —
     - —
     - 09/02/26
     - auto
     - `8 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/auto/2026-02-09_230708_46692/bn_error_3d_plot_initial.pdf>`__
     - `8 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/auto/2026-02-09_230708_46692/bn_error_3d_plot.pdf>`__
     - —
     - —
     - —
     - —
     - —
   * - 1.562
     - 5
     - 8
     - —
     - 2.9e-06
     - 7.7e-04
     - 2.8e-03
     - 2.0e+01
     - 3.4e-04
     - 8.0e-02
     - 2.2e-01
     - 2.1e+00
     - 8.1e+00
     - 2.4e+05
     - 7.9e+04
     - 3.1e+05
     - 1.0e+05
     - 0
     - 2.0e+02
     - 5.5e+00
     - —
     - —
     - 09/02/26
     - auto
     - `9 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/auto/2026-02-09_230708_30327/bn_error_3d_plot_initial.pdf>`__
     - `9 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/auto/2026-02-09_230708_30327/bn_error_3d_plot.pdf>`__
     - —
     - —
     - —
     - —
     - —
   * - 1.522
     - 7
     - 4
     - —
     - 1.6e-05
     - 1.8e-03
     - 7.8e-03
     - 2.0e+01
     - 7.4e-04
     - 1.1e-01
     - 1.5e-01
     - 2.6e+00
     - 1.0e+01
     - 1.4e+05
     - 2.8e+04
     - 2.1e+05
     - 5.8e+04
     - 0
     - 4.6e+02
     - 4.4e+00
     - —
     - —
     - 09/02/26
     - auto
     - `10 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/auto/2026-02-09_230708_24974/bn_error_3d_plot_initial.pdf>`__
     - `10 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/auto/2026-02-09_230708_24974/bn_error_3d_plot.pdf>`__
     - —
     - —
     - —
     - —
     - —
   * - 1.445
     - 6
     - 16
     - —
     - 7.5e-06
     - 1.3e-03
     - 4.8e-03
     - 2.0e+01
     - 3.2e-04
     - 7.9e-02
     - 2.1e-01
     - 2.6e+00
     - 1.0e+01
     - 1.7e+05
     - 4.7e+04
     - 1.9e+05
     - 6.3e+04
     - 0
     - 4.1e+02
     - 8.6e+00
     - 5.0e+02
     - 4.1e+02
     - 09/02/26
     - akaptano
     - `11 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/case/02-09-2026_20-19/bn_error_3d_plot_initial.pdf>`__
     - `11 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QA/akaptano/case/02-09-2026_20-19/bn_error_3d_plot.pdf>`__
     - —
     - —
     - —
     - —
     - —


.. _landremanpaul2021-qh-reactorscale-lowres:

Landreman-Paul QH
^^^^^^^^^^^^^^^^^

**Surface file:** ``LandremanPaul2021_QH_reactorScale_lowres``

This surface has 1 submission(s).
Typical configuration: 4 Fourier order, 5 base coils.

.. list-table:: Landreman-Paul QH Leaderboard
   :header-rows: 1
   :widths: auto

   * - :math:`\text{Score}`
     - :math:`N`
     - :math:`n`
     - :math:`\text{FC}`
     - :math:`f_{B}`
     - :math:`\bar{B}_n`
     - :math:`\max(B_n)`
     - :math:`L`
     - :math:`\mathrm{Var}(l_i)`
     - :math:`d_{cc}`
     - :math:`d_{cs}`
     - :math:`\bar{\kappa}`
     - :math:`MSC`
     - :math:`\bar{F}`
     - :math:`\bar{\tau}`
     - :math:`F_\text{max}`
     - :math:`\tau_\text{max}`
     - :math:`LN`
     - :math:`t`
     - :math:`\kappa_\text{max}`
     - :math:`\text{Date}`
     - :math:`\text{User}`
     - :math:`\text{i}`
     - :math:`\text{f}`
     - :math:`\text{PP}`
     - :math:`\text{BP}`
     - :math:`\text{QS}`
     - :math:`\text{iota}`
     - :math:`\text{FPT}`
   * - 1.159
     - 5
     - 4
     - —
     - 2.0e-02
     - 1.0e-03
     - 3.8e-03
     - 1.8e+02
     - 5.0e+00
     - 1.1e+00
     - 1.8e+00
     - 2.3e-01
     - 8.7e-02
     - 4.3e+07
     - 1.5e+08
     - 5.9e+07
     - 2.2e+08
     - 0
     - 4.7e+02
     - 6.8e-01
     - 08/02/26
     - akaptano
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QH_reactorScale_lowres/akaptano/basic_LandremanPaulQH/02-08-2026_12-50/bn_error_3d_plot_initial.pdf>`__
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QH_reactorScale_lowres/akaptano/basic_LandremanPaulQH/02-08-2026_12-50/bn_error_3d_plot.pdf>`__
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/LandremanPaul2021_QH_reactorScale_lowres/akaptano/basic_LandremanPaulQH/02-08-2026_12-50/poincare_plot.png>`__
     - —
     - —
     - —
     - —


.. _w7-x-without-coil-ripple-beta0p05-d23p4-tm:

W7-X
^^^^

**Surface file:** ``W7-X_without_coil_ripple_beta0p05_d23p4_tm``

This surface has 2 submission(s).
Typical configuration: 4 Fourier order, 4 base coils.

.. list-table:: W7-X Leaderboard
   :header-rows: 1
   :widths: auto

   * - :math:`\text{Score}`
     - :math:`N`
     - :math:`n`
     - :math:`\text{FC}`
     - :math:`f_{B}`
     - :math:`\bar{B}_n`
     - :math:`\max(B_n)`
     - :math:`L`
     - :math:`\mathrm{Var}(l_i)`
     - :math:`d_{cc}`
     - :math:`d_{cs}`
     - :math:`\bar{\kappa}`
     - :math:`MSC`
     - :math:`\bar{F}`
     - :math:`\bar{\tau}`
     - :math:`F_\text{max}`
     - :math:`\tau_\text{max}`
     - :math:`LN`
     - :math:`t`
     - :math:`\kappa_\text{max}`
     - :math:`\text{Date}`
     - :math:`\text{User}`
     - :math:`\text{i}`
     - :math:`\text{f}`
     - :math:`\text{PP}`
     - :math:`\text{BP}`
     - :math:`\text{QS}`
     - :math:`\text{iota}`
     - :math:`\text{FPT}`
   * - 1.088
     - 4
     - 4
     - 4,8,16
     - 8.1e-03
     - 3.9e-03
     - 2.2e-02
     - 4.5e+01
     - 1.3e-02
     - 2.2e-01
     - 3.3e-01
     - 1.2e+00
     - 1.8e+00
     - 3.8e+06
     - 3.8e+06
     - 6.0e+06
     - 6.6e+06
     - 0
     - 1.3e+03
     - 1.9e+00
     - 08/02/26
     - akaptano
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/W7-X_without_coil_ripple_beta0p05_d23p4_tm/akaptano/expert_W7X/02-08-2026_12-58/order_4/bn_error_3d_plot_initial.pdf>`__
     - `4 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/W7-X_without_coil_ripple_beta0p05_d23p4_tm/akaptano/expert_W7X/02-08-2026_12-58/order_4/bn_error_3d_plot.pdf>`__ `8 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/W7-X_without_coil_ripple_beta0p05_d23p4_tm/akaptano/expert_W7X/02-08-2026_12-58/order_8/bn_error_3d_plot.pdf>`__ `16 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/W7-X_without_coil_ripple_beta0p05_d23p4_tm/akaptano/expert_W7X/02-08-2026_12-58/order_16/bn_error_3d_plot.pdf>`__
     - —
     - —
     - —
     - —
     - —
   * - 1.027
     - 4
     - 4
     - 4,8
     - 7.9e-03
     - 3.8e-03
     - 2.4e-02
     - 4.5e+01
     - 2.4e-01
     - 2.4e-01
     - 3.5e-01
     - 1.1e+00
     - 1.8e+00
     - 3.6e+06
     - 3.2e+06
     - 4.6e+06
     - 4.9e+06
     - 0
     - 4.3e+02
     - 1.8e+00
     - 08/02/26
     - akaptano
     - `2 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/W7-X_without_coil_ripple_beta0p05_d23p4_tm/akaptano/basic_W7X/02-08-2026_12-57/order_4/bn_error_3d_plot_initial.pdf>`__
     - `4 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/W7-X_without_coil_ripple_beta0p05_d23p4_tm/akaptano/basic_W7X/02-08-2026_12-57/order_4/bn_error_3d_plot.pdf>`__ `8 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/W7-X_without_coil_ripple_beta0p05_d23p4_tm/akaptano/basic_W7X/02-08-2026_12-57/order_8/bn_error_3d_plot.pdf>`__
     - —
     - —
     - —
     - —
     - —


.. _c09r00-b-axis-half-tesla-ncsx-focus:

0.5 Tesla NCSX Design
^^^^^^^^^^^^^^^^^^^^^

**Surface file:** ``c09r00_B_axis_half_tesla_NCSX.focus``

This surface has 1 submission(s).
Typical configuration: 4 Fourier order, 4 base coils.

.. list-table:: 0.5 Tesla NCSX Design Leaderboard
   :header-rows: 1
   :widths: auto

   * - :math:`\text{Score}`
     - :math:`N`
     - :math:`n`
     - :math:`\text{FC}`
     - :math:`f_{B}`
     - :math:`\bar{B}_n`
     - :math:`\max(B_n)`
     - :math:`L`
     - :math:`\mathrm{Var}(l_i)`
     - :math:`d_{cc}`
     - :math:`d_{cs}`
     - :math:`\bar{\kappa}`
     - :math:`MSC`
     - :math:`\bar{F}`
     - :math:`\bar{\tau}`
     - :math:`F_\text{max}`
     - :math:`\tau_\text{max}`
     - :math:`LN`
     - :math:`t`
     - :math:`\kappa_\text{max}`
     - :math:`\text{Date}`
     - :math:`\text{User}`
     - :math:`\text{i}`
     - :math:`\text{f}`
     - :math:`\text{PP}`
     - :math:`\text{BP}`
     - :math:`\text{QS}`
     - :math:`\text{iota}`
     - :math:`\text{FPT}`
   * - 1.292
     - 4
     - 4
     - 4,8
     - 1.0e-04
     - 5.0e-03
     - 2.6e-02
     - 3.0e+01
     - 4.0e-03
     - 1.1e-01
     - 1.8e-01
     - 2.1e+00
     - 7.1e+00
     - 8.4e+04
     - 4.5e+04
     - 1.4e+05
     - 1.0e+05
     - 0
     - 3.4e+02
     - 4.0e+00
     - 08/02/26
     - akaptano
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/c09r00_B_axis_half_tesla_NCSX/akaptano/basic_NCSX/02-08-2026_12-50/order_4/bn_error_3d_plot_initial.pdf>`__
     - `4 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/c09r00_B_axis_half_tesla_NCSX/akaptano/basic_NCSX/02-08-2026_12-50/order_4/bn_error_3d_plot.pdf>`__ `8 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/c09r00_B_axis_half_tesla_NCSX/akaptano/basic_NCSX/02-08-2026_12-50/order_8/bn_error_3d_plot.pdf>`__
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/c09r00_B_axis_half_tesla_NCSX/akaptano/basic_NCSX/02-08-2026_12-50/poincare_plot.png>`__
     - —
     - —
     - —
     - —


.. _cfqs-2b40:

CFQS
^^^^

**Surface file:** ``cfqs_2b40``

This surface has 1 submission(s).
Typical configuration: 8 Fourier order, 4 base coils.

.. list-table:: CFQS Leaderboard
   :header-rows: 1
   :widths: auto

   * - :math:`\text{Score}`
     - :math:`N`
     - :math:`n`
     - :math:`\text{FC}`
     - :math:`f_{B}`
     - :math:`\bar{B}_n`
     - :math:`\max(B_n)`
     - :math:`L`
     - :math:`\mathrm{Var}(l_i)`
     - :math:`d_{cc}`
     - :math:`d_{cs}`
     - :math:`\bar{\kappa}`
     - :math:`MSC`
     - :math:`\bar{F}`
     - :math:`\bar{\tau}`
     - :math:`F_\text{max}`
     - :math:`\tau_\text{max}`
     - :math:`LN`
     - :math:`t`
     - :math:`\kappa_\text{max}`
     - :math:`\text{Date}`
     - :math:`\text{User}`
     - :math:`\text{i}`
     - :math:`\text{f}`
     - :math:`\text{PP}`
     - :math:`\text{BP}`
     - :math:`\text{QS}`
     - :math:`\text{iota}`
     - :math:`\text{FPT}`
   * - 1.461
     - 4
     - 8
     - —
     - 8.3e-05
     - 3.4e-03
     - 2.4e-02
     - 2.0e+01
     - 1.1e-03
     - 8.0e-02
     - 2.0e-01
     - 2.1e+00
     - 8.3e+00
     - 3.9e+05
     - 2.3e+05
     - 4.9e+05
     - 2.7e+05
     - 0
     - 1.6e+02
     - 5.1e+00
     - 08/02/26
     - akaptano
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/cfqs_2b40/akaptano/basic_CFQS/02-08-2026_12-50/bn_error_3d_plot_initial.pdf>`__
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/cfqs_2b40/akaptano/basic_CFQS/02-08-2026_12-50/bn_error_3d_plot.pdf>`__
     - —
     - —
     - —
     - —
     - —


.. _circular-tokamak:

Circular Tokamak
^^^^^^^^^^^^^^^^

**Surface file:** ``circular_tokamak``

This surface has 1 submission(s).
Typical configuration: 4 Fourier order, 6 base coils.

.. list-table:: Circular Tokamak Leaderboard
   :header-rows: 1
   :widths: auto

   * - :math:`\text{Score}`
     - :math:`N`
     - :math:`n`
     - :math:`\text{FC}`
     - :math:`f_{B}`
     - :math:`\bar{B}_n`
     - :math:`\max(B_n)`
     - :math:`L`
     - :math:`\mathrm{Var}(l_i)`
     - :math:`d_{cc}`
     - :math:`d_{cs}`
     - :math:`\bar{\kappa}`
     - :math:`MSC`
     - :math:`\bar{F}`
     - :math:`\bar{\tau}`
     - :math:`F_\text{max}`
     - :math:`\tau_\text{max}`
     - :math:`LN`
     - :math:`t`
     - :math:`\kappa_\text{max}`
     - :math:`\text{Date}`
     - :math:`\text{User}`
     - :math:`\text{i}`
     - :math:`\text{f}`
     - :math:`\text{PP}`
     - :math:`\text{BP}`
     - :math:`\text{QS}`
     - :math:`\text{iota}`
     - :math:`\text{FPT}`
   * - 2.006
     - 6
     - 4
     - —
     - 7.6e-03
     - 4.1e-03
     - 1.4e-02
     - 1.8e+02
     - 5.9e-03
     - 1.3e+00
     - 1.6e+00
     - 2.1e-01
     - 4.4e-02
     - 2.8e+06
     - 2.7e+05
     - 2.9e+06
     - 2.7e+05
     - 0
     - 3.1e+01
     - 2.2e-01
     - 08/02/26
     - akaptano
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/circular_tokamak/akaptano/basic_tokamak/02-08-2026_12-56/bn_error_3d_plot_initial.pdf>`__
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/circular_tokamak/akaptano/basic_tokamak/02-08-2026_12-56/bn_error_3d_plot.pdf>`__
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/circular_tokamak/akaptano/basic_tokamak/02-08-2026_12-56/poincare_plot.png>`__
     - —
     - —
     - —
     - —


.. _muse-focus:

MUSE
^^^^

**Surface file:** ``muse.focus``

This surface has 1 submission(s).
Typical configuration: 8 Fourier order, 4 base coils.

.. list-table:: MUSE Leaderboard
   :header-rows: 1
   :widths: auto

   * - :math:`\text{Score}`
     - :math:`N`
     - :math:`n`
     - :math:`\text{FC}`
     - :math:`f_{B}`
     - :math:`\bar{B}_n`
     - :math:`\max(B_n)`
     - :math:`L`
     - :math:`\mathrm{Var}(l_i)`
     - :math:`d_{cc}`
     - :math:`d_{cs}`
     - :math:`\bar{\kappa}`
     - :math:`MSC`
     - :math:`\bar{F}`
     - :math:`\bar{\tau}`
     - :math:`F_\text{max}`
     - :math:`\tau_\text{max}`
     - :math:`LN`
     - :math:`t`
     - :math:`\kappa_\text{max}`
     - :math:`\text{Date}`
     - :math:`\text{User}`
     - :math:`\text{i}`
     - :math:`\text{f}`
     - :math:`\text{PP}`
     - :math:`\text{BP}`
     - :math:`\text{QS}`
     - :math:`\text{iota}`
     - :math:`\text{FPT}`
   * - 1.632
     - 4
     - 8
     - —
     - 9.4e-07
     - 1.1e-02
     - 4.7e-02
     - 6.1e+00
     - 4.5e-06
     - 2.4e-02
     - 1.1e-01
     - 5.4e+00
     - 3.3e+01
     - 3.4e+03
     - 5.5e+02
     - 5.6e+03
     - 1.1e+03
     - 0
     - 2.6e+02
     - 1.0e+01
     - 08/02/26
     - akaptano
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/muse/akaptano/basic_MUSE/02-08-2026_12-50/bn_error_3d_plot_initial.pdf>`__
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/muse/akaptano/basic_MUSE/02-08-2026_12-50/bn_error_3d_plot.pdf>`__
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/muse/akaptano/basic_MUSE/02-08-2026_12-50/poincare_plot.png>`__
     - —
     - —
     - —
     - —


.. _wout-schuetthenneberg-nfp2-nc:

Schuett-Henneberg QA
^^^^^^^^^^^^^^^^^^^^

**Surface file:** ``wout_schuetthenneberg_nfp2.nc``

This surface has 1 submission(s).
Typical configuration: 8 Fourier order, 4 base coils.

.. list-table:: Schuett-Henneberg QA Leaderboard
   :header-rows: 1
   :widths: auto

   * - :math:`\text{Score}`
     - :math:`N`
     - :math:`n`
     - :math:`\text{FC}`
     - :math:`f_{B}`
     - :math:`\bar{B}_n`
     - :math:`\max(B_n)`
     - :math:`L`
     - :math:`\mathrm{Var}(l_i)`
     - :math:`d_{cc}`
     - :math:`d_{cs}`
     - :math:`\bar{\kappa}`
     - :math:`MSC`
     - :math:`\bar{F}`
     - :math:`\bar{\tau}`
     - :math:`F_\text{max}`
     - :math:`\tau_\text{max}`
     - :math:`LN`
     - :math:`t`
     - :math:`\kappa_\text{max}`
     - :math:`\text{Date}`
     - :math:`\text{User}`
     - :math:`\text{i}`
     - :math:`\text{f}`
     - :math:`\text{PP}`
     - :math:`\text{BP}`
     - :math:`\text{QS}`
     - :math:`\text{iota}`
     - :math:`\text{FPT}`
   * - 1.472
     - 4
     - 8
     - —
     - 2.8e-03
     - 5.1e-04
     - 3.5e-03
     - 1.5e+02
     - 2.6e-01
     - 3.9e-01
     - 1.3e+00
     - 3.4e-01
     - 2.0e-01
     - 7.0e+07
     - 2.3e+08
     - 7.7e+07
     - 2.7e+08
     - 0
     - 7.9e+01
     - 8.3e-01
     - 08/02/26
     - akaptano
     - `1 <https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main/submissions/wout_schuetthenneberg_nfp2/akaptano/basic_SchuettHennebergQA_nfp2/02-08-2026_12-55/bn_error_3d_plot_initial.pdf>`__
     - 1
     - —
     - —
     - —
     - —
     - —



.. note::
   Last updated: run ``stellcoilbench update-db`` to refresh locally.
