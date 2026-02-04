# Using MPI with StellCoilBench

StellCoilBench supports MPI parallelization for post-processing operations (VMEC and fieldline tracing), while coil optimization runs on a single core (rank 0).

## How It Works

When running with MPI:
- **Coil Optimization**: Runs only on **rank 0** (single core)
- **Post-Processing**: Uses **all MPI processes** for:
  - VMEC equilibrium calculations (via `MpiPartition`)
  - Fieldline tracing (via `comm_world`)
  - Other operations (QFM, plotting, etc.) run on rank 0 only

## Running with MPI

### Basic Usage

```bash
# Run with 4 MPI processes
mpirun -n 4 stellcoilbench submit-case cases/basic_MUSE.yaml

# Or with more control over CPU binding
mpirun -n 4 --bind-to core --map-by core stellcoilbench submit-case cases/basic_MUSE.yaml
```

### In CI Workflows

The self-hosted runner workflow (`.github/workflows/update-db-self-hosted.yml`) is configured to use 4 MPI processes per case:

```bash
mpirun -n 4 --bind-to core --map-by core python -u -m stellcoilbench.cli submit-case "$CASE_FILE"
```

With 16 parallel case runs, this uses:
- 16 cases × 4 MPI processes = 64 cores total

## Implementation Details

### Coil Optimization (Single Core)

The `optimize_coils()` function checks MPI rank:
- Only rank 0 runs the optimization loop
- Other ranks skip optimization and wait at a barrier
- After optimization completes, all ranks synchronize before post-processing

### Post-Processing (Multi-Core)

The `run_post_processing()` function automatically uses MPI:
- VMEC uses `MpiPartition(ngroups=1)` to use all processes
- Fieldline tracing uses `comm_world` for parallel tracing
- Non-parallel operations (QFM, plotting) run on rank 0 only

## Environment Variables

Set these to control threading (important when using MPI):

```bash
export OMP_NUM_THREADS=1          # Disable OpenMP threading
export MKL_NUM_THREADS=1           # Disable MKL threading
export OPENBLAS_NUM_THREADS=1      # Disable OpenBLAS threading
export VECLIB_MAXIMUM_THREADS=1    # Disable Accelerate threading (macOS)
```

These are automatically set in the CI workflow to prevent thread oversubscription.

## Performance Considerations

- **Coil Optimization**: Single-core (rank 0 only) - typically 1-3 minutes
- **Post-Processing**: Multi-core MPI parallelization:
  - VMEC: Scales well with MPI processes
  - Fieldline tracing: Scales well with number of fieldlines
  - Other operations: Single-core (negligible time)

## Troubleshooting

### MPI Not Detected

If MPI is not detected, the code runs in single-process mode:
- Check that `mpi4py` is installed: `pip install mpi4py`
- Check that `simsopt` was built with MPI support
- Verify MPI is available: `mpirun --version`

### Processes Hanging

If processes hang at barriers:
- Ensure all processes reach the barrier (check for errors on rank 0)
- Check MPI communication: `mpirun -n 4 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.rank)"`

### Performance Issues

- Ensure `OMP_NUM_THREADS=1` to avoid thread oversubscription
- Check CPU binding with `--bind-to core --map-by core`
- Monitor CPU usage: should see rank 0 at 100% during optimization, all ranks active during VMEC
