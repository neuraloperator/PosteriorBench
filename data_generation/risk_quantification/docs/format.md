# Data and parameter format

The dataset uses schema version 2 and stores risk probability in the single field
`u`. HF field arrays are float32. Parameter coefficients retain float64 precision.

## Meaning of the fields

The scalar state follows `dX=b(X,t) dt + sqrt(2 D(X,t)) dW`. The safe set C(t) is
a union of open intervals. Reaching a boundary is an exit. `u(tau,x)` is exit risk
by deadline 1, with chronological start time `t=1-tau`. Its axes are remaining time
and position; chronological time is not increasing along the array's first axis.
At tau=0, risk is zero inside C(1) and one outside; that row is excluded from observations.

Grids are uniformly spaced on tau=[0,1] and the saved physical domain, including
endpoints. Config specifies nx, nt, internal solver resolution nz, substeps,
integrator, domain, prior dimensions, seed, and stream. HF case metadata also
stores explicit grids, observed values, and their coordinates.

## Parameter arrays

Let N be the number of samples, J=max_components, and K=spatial_modes.

| Key | Shape | Meaning |
|---|---|---|
| sample_id | N | Sample identity within the recorded stream |
| ncomp | N | Number of active safe intervals |
| full_domain | N | Full-domain flag; zero for moving_mix |
| geometry | N,2J+1,5 | Time-basis coefficients for interval and gap geometry |
| beta | N,J,3,3+2K | Component-local drift coefficients |
| logdiff | N,J,3,3+2K | Component-local log-diffusivity coefficients |
| frequency | N,J | Dynamics time frequencies |
| clock | N,5 | Log time-change coefficients |
| tracking | N,J | Tracking of moving boundaries |
| family | N,J | Dynamics family labels |
| geometry_family | N | Geometry family label |

The label definitions are `risk1d.FAMILY_NAMES` and `safe_explorer_core.FAMILY_NAMES`.
Use ncomp to ignore padded components. The saved coefficients are not dense
physical drift or diffusivity: `risk1d.evaluate_settings(params,cfg)` applies the
coordinate transforms and returns safe intervals, safe_mask, drift, diffusivity,
and noise_std=sqrt(2D). It takes **chronological** times. Dynamics outside the safe
set are NaN, because they do not affect the killed process.

## File mapping

Pool NPZ blocks contain `u`, `sample_ids`, and `parameter__*` arrays in matching row
order. The pool manifest gives relative paths, ID ranges, checksums, and Config.
Training parameter NPZ files contain the parameter names directly. Case parameter
files contain `reference__*` and `posterior__*` arrays. The latter match the order
of `posterior_samples_u` and `posterior_weights` exactly.

The benchmark manifest's case_index maps each case to its split and row, original
reference-bank ID, reference/candidate streams, posterior IDs, and parameter file.
The case number is not the original reference-bank ID. Numeric IDs alone are not
globally unique: interpret them with their stream and configuration.

HF training is one Dataset with `u` and `metadata`. HF posterior data is a
DatasetDict with validation/test splits and columns `u_ref`, `posterior_samples_u`,
`posterior_weights`, and `metadata`. Weights sum to one within each case. Metadata
includes observed values/locations, grids, K, RMSE errors, and sample provenance.

Parameters in the existing distribution were reconstructed and checked against
saved originals. Newly generated pools save the actual parameters passed to the
solver. The manifest records the origin; the loader uses the saved arrays in both
cases. No random generator or PDE solver is needed for loading or visualization.

Only observed values and locations are conditioning inputs. Full reference fields,
posterior fields, and hidden parameters belong to evaluation. The independent
training pool has a separate stream from case references and posterior candidates.
