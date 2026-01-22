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
     
     - ``ftol``: Function tolerance (default: 1e-6)
     - ``gtol``: Gradient tolerance (default: 1e-5)
     - ``maxfun``: Maximum function evaluations
     - ``maxcor``: Maximum number of variable metric corrections (L-BFGS-B)

**Objective Terms**
   
   The ``coil_objective_terms`` section defines which terms to include in the objective
   function and how to penalize them. Each term can use different penalty types:
   
   - ``l1``: L1 norm penalty
   - ``l1_threshold``: Thresholded L1 penalty (only penalizes values above threshold)
   - ``l2``: L2 norm penalty
   - ``l2_threshold``: Thresholded L2 penalty (only penalizes values above threshold)
   - ``lp``: Lp norm penalty (requires corresponding ``*_p`` parameter)
   - ``lp_threshold``: Thresholded Lp penalty (requires corresponding ``*_p`` parameter)
   - ``""``: Include term without penalty (for linking number)
   
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
       ftol: 1e-6
       gtol: 1e-5
   
   coil_objective_terms:
     total_length: "l2_threshold"
     coil_coil_distance: "l1_threshold"
     coil_surface_distance: "l1_threshold"
     coil_curvature: "lp_threshold"
     coil_curvature_p: 2
     coil_mean_squared_curvature: "l2_threshold"
     coil_arclength_variation: "l2_threshold"
     linking_number: ""
     coil_coil_force: "lp_threshold"
     coil_coil_force_p: 2
     coil_coil_torque: "lp_threshold"
     coil_coil_torque_p: 2

Available Objective Terms
-------------------------

**total_length**
   Penalizes the total length of all coils.
   
   Options: ``l1``, ``l1_threshold``, ``l2``, ``l2_threshold``
   
   Mathematical form:
   
   .. math::
      L = \sum_{i=1}^{N} \int_{0}^{L_i} ds
   
   Thresholded versions only penalize lengths above a threshold, allowing coils to
   be shorter than the threshold without penalty.

**coil_coil_distance**
   Penalizes distances between coils to prevent collisions.
   
   Options: ``l1``, ``l1_threshold``, ``l2``, ``l2_threshold``
   
   Measures the minimum distance between any two coils. Thresholded versions ensure
   coils maintain a minimum separation distance.

**coil_surface_distance**
   Penalizes distances between coils and the plasma surface.
   
   Options: ``l1``, ``l1_threshold``, ``l2``, ``l2_threshold``
   
   Ensures coils maintain a safe distance from the plasma surface. Too close and
   coils may interfere with plasma operation; too far and field quality degrades.

**coil_curvature**
   Penalizes coil curvature to ensure manufacturability.
   
   Options: ``lp``, ``lp_threshold`` (requires ``coil_curvature_p``)
   
   Curvature is defined as :math:`\kappa = |\mathbf{r}''(s)|`. The Lp norm is:
   
   .. math::
      \left( \int \kappa^p ds \right)^{1/p}
   
   Typical values for ``coil_curvature_p``: 2 (L2 norm) or higher for stronger
   penalties on high curvature regions.

**coil_mean_squared_curvature**
   Penalizes mean squared curvature across coils.
   
   Options: ``l1``, ``l1_threshold``, ``l2``, ``l2_threshold``
   
   Mathematical form:
   
   .. math::
      \text{MSC} = \frac{1}{N} \sum_{i=1}^{N} \int \kappa_i^2 ds
   
   This provides a smoother penalty than maximum curvature, encouraging overall
   smoothness rather than just avoiding extreme values.

**coil_arclength_variation**
   Penalizes variation in arclength between coil segments.
   
   Options: ``l2_threshold``
   
   Ensures coils have uniform spacing, which is important for manufacturing
   and field quality. The threshold is scaled by :math:`R_0^2` where :math:`R_0`
   is the major radius of the plasma surface.

**linking_number**
   Constrains the topological linking number between coils.
   
   Options: ``""`` (empty string to include, omit key to exclude)
   
   The linking number measures how coils are topologically linked:
   
   .. math::
      \text{LN} = \frac{1}{4\pi} \sum_{i \neq j} \oint_{C_i} \oint_{C_j} \frac{(\mathbf{r}_i - \mathbf{r}_j) \cdot (d\mathbf{r}_i \times d\mathbf{r}_j)}{|\mathbf{r}_i - \mathbf{r}_j|^3}
   
   Including this term ensures coils maintain their topological structure during
   optimization.

**coil_coil_force**
   Penalizes forces between coils.
   
   Options: ``lp``, ``lp_threshold`` (requires ``coil_coil_force_p``)
   
   Forces arise from electromagnetic interactions between coils. High forces
   indicate coils that may be difficult to support mechanically.

**coil_coil_torque**
   Penalizes torques between coils.
   
   Options: ``lp``, ``lp_threshold`` (requires ``coil_coil_torque_p``)
   
   Torques indicate rotational forces that must be resisted by coil supports.
   High torques can lead to mechanical instability.

Penalty Types Explained
-----------------------

**L1 Norm**
   Sum of absolute values: :math:`\sum_i |x_i|`
   
   Encourages sparsity (many values become exactly zero). Useful for distance
   penalties where you want to avoid any violations.

**L2 Norm**
   Square root of sum of squares: :math:`\sqrt{\sum_i x_i^2}`
   
   Provides smooth penalties. Useful for length and curvature penalties where
   you want gradual increases in penalty with violation magnitude.

**Lp Norm**
   Generalization: :math:`\left(\sum_i |x_i|^p\right)^{1/p}`
   
   Allows tuning the penalty strength. Higher :math:`p` gives stronger penalties
   for large violations. Typical values: 2 (L2), 4, 8.

**Thresholded Penalties**
   Only penalize values above a threshold:
   
   .. math::
      \text{penalty} = \begin{cases}
         0 & \text{if } x \leq \text{threshold} \\
         \text{norm}(x - \text{threshold}) & \text{if } x > \text{threshold}
      \end{cases}
   
   This allows values below the threshold without penalty, which is useful for
   engineering constraints (e.g., minimum separation distances).

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
        coil_coil_distance: "l1_threshold"
        coil_surface_distance: "l1_threshold"

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
        coil_coil_distance: "l1_threshold"
        coil_surface_distance: "l1_threshold"
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
          ftol: 1e-8
          gtol: 1e-7
      coil_objective_terms:
        total_length: "l2_threshold"
        coil_coil_distance: "l1_threshold"
        coil_surface_distance: "l1_threshold"
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
