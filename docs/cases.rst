Cases
=====

Case files (YAML) define the optimization problem: plasma surface, coil configuration, optimizer, and objective terms.

Schema
------

**surface_params**
   - ``surface``: Name from ``plasma_surfaces/`` (e.g. ``input.LandremanPaul2021_QA``)
   - ``range``: ``"half period"``, ``"full period"``, or ``"full torus"``

**coils_params**
   - ``ncoils``: Number of coils (4, 6, 8, …)
   - ``order``: Fourier order (4, 8, 16, …)

**optimizer_params**
   - ``algorithm``: ``"L-BFGS-B"`` (recommended) or ``"augmented_lagrangian"``
   - ``max_iterations``: e.g. 200–1000
   - ``max_iter_subopt``: For augmented Lagrangian (e.g. 10–40)

**coil_objective_terms**
   Each term maps to a penalty type: ``l1``, ``l2``, ``lp``, ``l1_threshold``, ``l2_threshold``, ``lp_threshold``, or ``""``.
   Common terms: ``total_length``, ``coil_curvature``, ``coil_mean_squared_curvature``, ``coil_arclength_variation``, ``linking_number``, ``coil_coil_force``, ``coil_coil_torque``.
   ``coil_coil_distance`` and ``coil_surface_distance`` are always included; use ``cc_threshold`` and ``cs_threshold``.

**fourier_continuation** (optional)
   Progressive refinement by order: ``enabled: true``, ``orders: [4, 8, 16]``.

Example
-------

.. code-block:: yaml

   description: "Basic Landreman-Paul QA case"
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

See :doc:`leaderboard/metric_definitions` for metric notation and :doc:`api` for full schema details.
