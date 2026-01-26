Cases
=====

Case files define the optimization problem for StellCoilBench. Each case is a YAML file
that specifies the plasma surface, coil configuration, optimization algorithm, and objective
function terms.

Case File Structure
-------------------

Each case file contains four main sections:

1. **``description``**: Free-text description of the case
2. **``surface_params``**: Plasma surface configuration
3. **``coils_params``**: Coil geometry parameters
4. **``optimizer_params``**: Optimization algorithm and settings
5. **``coil_objective_terms``**: Objective function terms and their penalty types

Schema Overview
---------------

**Description**
   A human-readable description of the case. This helps identify the purpose and
   characteristics of each benchmark.

**Surface Parameters**
   
   - ``surface``: Name of the plasma surface file (without extension)
   
     Available surfaces are in ``plasma_surfaces/``. Examples:
     
     - ``input.LandremanPaul2021_QA``
     - ``input.circular_tokamak``
     - ``input.W7-X_without_coil_ripple_beta0p05_d23p4_tm``
     - ``input.HSX_QHS_mn1824_ns101``
     - ``input.cfqs_2b40``
     - ``input.rotating_ellipse``
     - ``c09r00_B_axis_half_tesla_PM4Stell.focus``
     - ``muse.focus``
   
   - ``range``: Surface range to use
     
     Options:
     
     - ``"half period"``: Use half-period symmetry (default for stellarators)
     - ``"full period"``: Use full period
     - ``"full torus"``: Use full torus (no symmetry)

**Coil Parameters**
   
   - ``ncoils``: Number of base coils (before applying stellarator symmetry)
     
     Typical values: 4, 6, 8, 12. More coils allow more complex field shaping but
     increase computational cost.
   
   - ``order``: Fourier order for coil representation
     
     Coils are represented as Fourier series: :math:`\mathbf{r}(\phi) = \sum_{m=-n}^{n} \mathbf{c}_m e^{im\phi}`
     
     Higher order allows more complex coil shapes but increases the number of
     optimization variables. Typical values: 4, 8, 16.
     
     **Note**: This is the final order used if Fourier continuation is disabled.
     If Fourier continuation is enabled, this value is ignored in favor of the
     continuation schedule (see :ref:`fourier-continuation`).

**Optimizer Parameters**
   
   - ``algorithm``: Optimization algorithm to use
     
     Supported algorithms:
     
     - ``"L-BFGS-B"``: Limited-memory BFGS with bounds (recommended for most cases)
     - ``"augmented_lagrangian"``: Augmented Lagrangian method for constrained optimization
     - Any algorithm supported by ``scipy.optimize.minimize``
   
   - ``max_iterations``: Maximum number of optimization iterations
     
     Typical values: 100-500 for testing, 1000+ for production runs.
   
   - ``max_iter_subopt``: Maximum iterations for sub-optimization
     
     Used by augmented Lagrangian method. Typical values: 10-40.
   
   - ``verbose``: Print optimization progress
     
     Set to ``True`` for debugging, ``False`` for quiet runs.
   
   - ``algorithm_options``: Algorithm-specific options
     
     Dictionary of options passed to the optimizer. Common options:
     
     - ``ftol``: Function tolerance (default: 1e-12 for L-BFGS-B)
     - ``gtol``: Gradient tolerance (default: 1e-12 for L-BFGS-B)
     - ``tol``: General tolerance (default: 1e-12 for L-BFGS-B)
     - ``maxfun``: Maximum function evaluations
     - ``maxcor``: Maximum number of variable metric corrections (L-BFGS-B)
     
     **Note**: For L-BFGS-B, the default tolerances are set to 1e-12 (ftol, gtol, tol)
     for stricter convergence. For other algorithms, scipy defaults apply.

**Objective Terms**
   
   The ``coil_objective_terms`` section defines which terms to include in the objective
   function and how to penalize them. Each term can use different penalty types:
   
   - ``l1``: L1 norm penalty
   - ``l1_threshold``: Thresholded L1 penalty (only penalizes values above threshold)
   - ``l2``: L2 norm penalty
   - ``l2_threshold``: Thresholded L2 penalty (only penalizes values above threshold)
   - ``lp``: Lp norm penalty (requires corresponding ``*_p`` parameter)
   - ``lp_threshold``: Thresholded Lp penalty (requires corresponding ``*_p`` parameter)
   - ``""``: Include term without penalty (for linking number, or optionally for coil_coil_distance/coil_surface_distance)
   
   .. note::
      ``coil_coil_distance`` and ``coil_surface_distance`` are always included automatically. 
      You can optionally specify them with ``""``, but this is not required. Use ``cc_threshold``/``cs_threshold`` 
      for thresholds and ``constraint_weight_1``/``constraint_weight_2`` for weights.
   
   **Note**: ``SquaredFlux`` is always included in the objective function and cannot
   be excluded.

Example Case File
-----------------

Here's a complete example case file:

.. code-block:: yaml

   description: "Basic test case with Landreman-Paul QA surface"
   surface_params:
     surface: "input.LandremanPaul2021_QA"
     range: "half period"
   
   coils_params:
     ncoils: 4
     order: 4
   
   optimizer_params:
     algorithm: "L-BFGS-B"
     max_iterations: 200
     max_iter_subopt: 10
     verbose: False
     algorithm_options:
       ftol: 1e-12
       gtol: 1e-12
   
   coil_objective_terms:
     total_length: "l2_threshold"
     coil_curvature: "lp_threshold"
     coil_curvature_p: 2
     coil_mean_squared_curvature: "l2_threshold"
     coil_arclength_variation: "l2_threshold"
     linking_number: ""
     coil_coil_force: "lp_threshold"
     coil_coil_force_p: 2
     coil_coil_torque: "lp_threshold"
     coil_coil_torque_p: 2

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

Available Objective Terms
-------------------------

**total_length**
   Penalizes the total length of all coils.
   
   Options: ``l1``, ``l1_threshold``, ``l2``, ``l2_threshold``
   
   Mathematical form:
   
   .. math::
      L = \sum_{i=1}^{N} \int_{C_i} d\ell_i
   
   Units: :math:`\text{m}` (meters)
   
   For thresholded versions:
   
   - L1 thresholded: :math:`\max\left(L - L_0, 0\right)` where :math:`L_0` is the desired length threshold
   - L2 thresholded: :math:`\max\left(L - L_0, 0\right)^2`
   
   This allows coils to be shorter than the threshold without penalty.

**coil_coil_distance**
   Penalizes distances between coils to prevent collisions.
   
   .. note::
      This objective is **always included automatically**. ``CurveCurveDistance`` computes squared penalties 
      with threshold internally. You can optionally specify it with an empty string ``""`` in 
      ``coil_objective_terms``, but this is not required. Use ``cc_threshold`` to set the threshold 
      and ``constraint_weight_1`` to set the weight.
   
   Mathematical form:
   
   .. math::
      J = \sum_{i = 1}^{N} \sum_{j = 1}^{i-1} d_{i,j}
   
   where
   
   .. math::
      d_{i,j} = \int_{C_i} \int_{C_j} \max\left(0, d_{\min} - \left\| \mathbf{r}_i - \mathbf{r}_j \right\|_2\right)^2 ~d\ell_j ~d\ell_i
   
   and :math:`\mathbf{r}_i`, :math:`\mathbf{r}_j` are points on coil curves :math:`C_i` and :math:`C_j`, respectively, and :math:`d_\min` is the desired threshold minimum intercoil distance (specified via ``cc_threshold``).
   
   Units: :math:`\text{m}^2` (meters squared)
   
   This ensures coils maintain a minimum separation distance.

**coil_surface_distance**
   Penalizes distances between coils and the plasma surface.
   
   .. note::
      This objective is **always included automatically**. ``CurveSurfaceDistance`` computes squared penalties 
      with threshold internally. You can optionally specify it with an empty string ``""`` in 
      ``coil_objective_terms``, but this is not required. Use ``cs_threshold`` to set the threshold 
      and ``constraint_weight_2`` to set the weight.
   
   Mathematical form:
   
   .. math::
      J = \sum_{i = 1}^{N} d_{i}
   
   where
   
   .. math::
      d_{i} = \int_{C_i} \int_{S} \max\left(0, d_{\min} - \left\| \mathbf{r}_i - \mathbf{s} \right\|_2\right)^2 ~d\ell_i ~ds
   
   and :math:`\mathbf{r}_i` is a point on coil curve :math:`C_i` and :math:`\mathbf{s}` is a point on the plasma surface :math:`S`, and :math:`d_\min` is the desired threshold minimum coil-to-surface distance (specified via ``cs_threshold``).
   
   Units: :math:`\text{m}^2` (meters squared)
   
   Ensures coils maintain a safe distance from the plasma surface.

**coil_curvature**
   Penalizes coil curvature to ensure manufacturability.
   
   Options: ``lp``, ``lp_threshold`` (requires ``coil_curvature_p``)
   
   Curvature is defined as :math:`\kappa_i(\ell_i) = \left|\mathbf{r}_i''(\ell_i)\right|` where :math:`\mathbf{r}_i(\ell_i)` is the parameterization of coil curve :math:`C_i` by arclength.
   
   Units: :math:`\text{m}^{-1}` (inverse meters)
   
   Mathematical form (per coil):
   
   .. math::
      \frac{1}{p} \int_{C_i} \max\left(\kappa_i - \kappa_0, 0\right)^p ~d\ell_i
   
   For ``lp`` option, :math:`\kappa_0 = 0`, so this becomes :math:`\frac{1}{p} \int_{C_i} \kappa_i^p ~d\ell_i`.
   
   For ``lp_threshold`` option, :math:`\kappa_0` is the desired maximum curvature threshold. Typical values for ``coil_curvature_p``: 2 (L2 norm) or higher for stronger penalties on high curvature regions.

**coil_mean_squared_curvature**
   Penalizes mean squared curvature across coils.
   
   Options: ``l1``, ``l1_threshold``, ``l2``, ``l2_threshold``
   
   Mathematical form (per coil):
   
   .. math::
      J = \frac{1}{L_i} \int_{C_i} \kappa_i^2 ~d\ell_i
   
   where :math:`L_i` is the total length of coil curve :math:`C_i`, :math:`\ell_i` is the arclength along the curve, and :math:`\kappa_i` is the curvature.
   
   Units: :math:`\text{m}^{-2}` (inverse meters squared)
   
   For thresholded versions:
   
   - L1 thresholded: :math:`\max\left(J - J_0, 0\right)` where :math:`J_0` is the desired threshold
   - L2 thresholded: :math:`\max\left(J - J_0, 0\right)^2`
   
   This provides a smoother penalty than maximum curvature, encouraging overall smoothness rather than just avoiding extreme values.

**coil_arclength_variation**
   Penalizes variation in arclength between coil segments.
   
   Options: ``l2_threshold``
   
   Measures the variance of incremental arclength :math:`J = \text{Var}(l_i)` where :math:`l_i` is the average incremental arclength on interval :math:`I_i` from a partition :math:`\{I_i\}_{i=1}^L` of :math:`[0,1]`.
   
   Units: :math:`\text{m}^2` (meters squared)
   
   For L2 thresholded:
   
   .. math::
      \max\left(J - J_0, 0\right)^2
   
   where :math:`J_0` is the desired threshold (scaled by :math:`R_0^2` where :math:`R_0` is the major radius). Ensures coils have uniform spacing, which is important for manufacturing and field quality.

**linking_number**
   Constrains the topological linking number between coils.
   
   Options: ``""`` (empty string to include, omit key to exclude)
   
   The linking number measures how coils are topologically linked:
   
   .. math::
      \text{LN} = \frac{1}{4\pi} \sum_{i \neq j} \oint_{C_i} \oint_{C_j} \frac{\left(\mathbf{r}_i - \mathbf{r}_j\right) \cdot \left(d\mathbf{r}_i \times d\mathbf{r}_j\right)}{\left|\mathbf{r}_i - \mathbf{r}_j\right|^3}
   
   Units: dimensionless
   
   Including this term ensures coils maintain their topological structure during
   optimization.

**coil_coil_force**
   Penalizes forces between coils.
   
   Options: ``lp``, ``lp_threshold`` (requires ``coil_coil_force_p``)
   
   Forces :math:`\frac{d\vec{F}_i}{d\ell_i}` (Lorentz force per unit length) arise from electromagnetic interactions between coils. The Lp norm is:
   
   .. math::
      \frac{1}{p}\sum_{i=1}^{N}\frac{1}{L_i}\left(\int_{C_i} \left|\frac{d\vec{F}_i}{d\ell_i}\right|^p d\ell_i\right)
   
   where :math:`L_i` is the total length of coil curve :math:`C_i`.
   
   Units: :math:`\left(\text{N}/\text{m}\right)^p` (Newtons per meter to the power p)
   
   For Lp thresholded:
   
   .. math::
      \frac{1}{p}\sum_{i=1}^{N}\frac{1}{L_i}\left(\int_{C_i} \max\left(\left|\frac{d\vec{F}_i}{d\ell_i}\right| - \frac{dF_0}{d\ell_i}, 0\right)^p d\ell_i\right)
   
   where :math:`\frac{dF_0}{d\ell_i}` is the desired maximum force threshold per unit length. High forces indicate coils that may be difficult to support mechanically.

**coil_coil_torque**
   Penalizes torques between coils.
   
   Options: ``lp``, ``lp_threshold`` (requires ``coil_coil_torque_p``)
   
   Torques :math:`\frac{d\vec{T}_i}{d\ell_i}` (Lorentz torque per unit length) indicate rotational forces that must be resisted by coil supports. The Lp norm is:
   
   .. math::
      \frac{1}{p}\sum_{i=1}^{N}\frac{1}{L_i}\left(\int_{C_i} \left|\frac{d\vec{T}_i}{d\ell_i}\right|^p d\ell_i\right)
   
   where :math:`L_i` is the total length of coil curve :math:`C_i`.
   
   Units: :math:`\text{N}^p` (Newtons to the power p)
   
   For Lp thresholded:
   
   .. math::
      \frac{1}{p}\sum_{i=1}^{N}\frac{1}{L_i}\left(\int_{C_i} \max\left(\left|\frac{d\vec{T}_i}{d\ell_i}\right| - \frac{dT_0}{d\ell_i}, 0\right)^p d\ell_i\right)
   
   where :math:`\frac{dT_0}{d\ell_i}` is the desired maximum torque threshold per unit length. High torques can lead to mechanical instability.

Penalty Types Explained
-----------------------

**L1 Norm**
   Sum of absolute values: :math:`\sum_i \left|x_i\right|`
   
   Encourages sparsity (many values become exactly zero). Useful for distance
   penalties where you want to avoid any violations.

**L2 Norm**
   Square root of sum of squares: :math:`\sqrt{\sum_i x_i^2}`
   
   Provides smooth penalties. Useful for length and curvature penalties where
   you want gradual increases in penalty with violation magnitude.

**Lp Norm**
   Generalization: :math:`\left(\sum_i \left|x_i\right|^p\right)^{1/p}`
   
   Allows tuning the penalty strength. Higher :math:`p` gives stronger penalties
   for large violations. Typical values: 2 (L2), 4, 8.

**Thresholded Penalties**
   Thresholded penalties only penalize values above a desired threshold :math:`x_0`:
   
   - **L1 thresholded**: :math:`\max\left(x - x_0, 0\right)` for discrete quantities, or :math:`\int \max\left(x - x_0, 0\right) \, d\ell` for continuous quantities
   
   - **L2 thresholded**: :math:`\max\left(x - x_0, 0\right)^2` for discrete quantities, or :math:`\int \max\left(x - x_0, 0\right)^2 \, d\ell` for continuous quantities
   
   - **Lp thresholded**: :math:`\frac{1}{p} \int \max\left(x - x_0, 0\right)^p \, d\ell` for continuous quantities (e.g., curvature, force, torque)
   
   This allows values below the threshold without penalty, which is useful for
   engineering constraints (e.g., minimum separation distances, maximum curvature limits).

Case File Validation
--------------------

Validate case files before running:

.. code-block:: bash

   stellcoilbench validate-config cases/your_case.yaml

This checks for:
- Required fields
- Valid surface names
- Valid algorithm names
- Valid objective term options
- Algorithm-specific option compatibility

Common Case Configurations
---------------------------

**Basic Case**
   Minimal configuration for quick testing:
   
   .. code-block:: yaml
   
      description: "Basic test"
      surface_params:
        surface: "input.circular_tokamak"
        range: "half period"
      coils_params:
        ncoils: 4
        order: 4
      optimizer_params:
        algorithm: "L-BFGS-B"
        max_iterations: 100
        max_iter_subopt: 10
        verbose: False
      coil_objective_terms:
        total_length: "l2_threshold"
        coil_coil_distance: ""  # CurveCurveDistance already handles thresholding and squaring internally
        coil_surface_distance: ""  # CurveSurfaceDistance already handles thresholding and squaring internally

**Expert Case**
   Full configuration with all terms:
   
   .. code-block:: yaml
   
      description: "Expert configuration"
      surface_params:
        surface: "input.LandremanPaul2021_QA"
        range: "half period"
      coils_params:
        ncoils: 4
        order: 16
      optimizer_params:
        algorithm: "augmented_lagrangian"
        max_iterations: 500
        max_iter_subopt: 40
        verbose: True
      coil_objective_terms:
        total_length: "l2_threshold"
        coil_coil_distance: ""  # CurveCurveDistance already handles thresholding and squaring internally
        coil_surface_distance: ""  # CurveSurfaceDistance already handles thresholding and squaring internally
        coil_curvature: "lp_threshold"
        coil_curvature_p: 2
        coil_mean_squared_curvature: "l2_threshold"
        coil_arclength_variation: "l2_threshold"
        linking_number: ""
        coil_coil_force: "lp_threshold"
        coil_coil_force_p: 2

**High-Resolution Case**
   For production-quality results:
   
   .. code-block:: yaml
   
      description: "High-resolution optimization"
      surface_params:
        surface: "input.LandremanPaul2021_QA"
        range: "half period"
      coils_params:
        ncoils: 6
        order: 16
      optimizer_params:
        algorithm: "L-BFGS-B"
        max_iterations: 1000
        max_iter_subopt: 10
        verbose: False
        algorithm_options:
          ftol: 1e-12
          gtol: 1e-12
      coil_objective_terms:
        total_length: "l2_threshold"
        coil_coil_distance: ""  # CurveCurveDistance already handles thresholding and squaring internally
        coil_surface_distance: ""  # CurveSurfaceDistance already handles thresholding and squaring internally
        coil_curvature: "lp_threshold"
        coil_curvature_p: 4
        coil_mean_squared_curvature: "l2_threshold"

Best Practices
--------------

1. **Start Simple**: Begin with basic cases (fewer objective terms, lower order)
   and gradually add complexity.

2. **Use Thresholded Penalties**: Thresholded penalties are often more effective
   than pure norms for engineering constraints.

3. **Tune Algorithm Options**: Adjust ``ftol`` and ``gtol`` based on your
   convergence requirements. Tighter tolerances improve quality but increase
   computation time.

4. **Monitor Optimization**: Use ``verbose: True`` during development to
   understand optimization behavior.

5. **Validate Before Running**: Always validate case files to catch errors early.

6. **Document Your Cases**: Use descriptive ``description`` fields to explain
   the purpose and characteristics of each case.
