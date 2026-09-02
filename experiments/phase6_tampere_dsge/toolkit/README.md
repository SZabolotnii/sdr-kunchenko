# Vendored DSGE toolkit modules

`run.py`, `plot.py` and the scripts under `iterations/` import four modules that live in
the author's separate DSGE toolkit, not in this package:

| File | Role |
|---|---|
| `basis_functions.py` | generating-element basis construction |
| `kunchenko_features.py` | real-valued DSGE reconstruction-error features |
| `complex_features.py` | complex I/Q wrappers (`magnitude_phase`, `re_im`, `analytic`) |
| `splits.py` | leakage-safe grouped splits and the leakage audit |
| `verify_dsge.py` | cluster bootstrap comparison, PATP boundary check, report writer |

They are vendored here, unmodified, so that this package is self-contained: in the
research tree they were reached through an absolute path that only existed on the machine
where the experiments were run, which made the pipeline unrunnable elsewhere. The only
change made in this repository is to the *importing* scripts, which now resolve this
directory relative to their own location.

Same author, same MIT license as the rest of this repository.
