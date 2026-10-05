# Resonator Fits — web app

Browse QCoDeS measurement databases, plot amplitude and phase, and fit resonances
(f_r, Q_L, Q_i, Q_c) with the circle-fit routines from
[RFP-Software](https://github.com/ign4si/RFP-Software) (`resonator_tools`).

## Run

```bash
pip install -r requirements.txt     # fastapi, uvicorn, numpy, scipy, matplotlib
./run.sh                            # scans ../Database for *.db  ->  http://127.0.0.1:8050
./run.sh --db "/path/to/Database"   # any directory (searched recursively) or a single .db
./run.sh --host 0.0.0.0             # reachable from other machines in the lab
```

The database is opened **read-only**; nothing is ever written to it. The path can
also be set with the `RESONATOR_DB` environment variable.

## Using it

* **Pick** database → experiment → run in the sidebar (↑/↓ steps through runs; the zoom is kept
  when the new run covers the same span, handy for following one resonance through a power sweep).
* **Manual fit:** with *Select range* active (key `S`), drag horizontally across a resonance on
  either plot. It is fitted immediately (untick *fit on select* to only fill f₁/f₂). You can also type
  f₁/f₂ in GHz, or press *Use view* to fit the visible range. `Z` = zoom, `P` = pan, `R` = reset.
* **Auto:** *Detect* marks the dips; *Detect & fit all* also fits every one. Click a row to zoom to it,
  ↻ to refit with the current options, ✕ to drop a bad one.
* **Batch:** *Fit all runs* takes the fit windows of the current trace and fits them in every run of the
  experiment (each window is recentred on the local minimum, so small drifts are followed). Plot Qi, Qc, QL,
  f_r shift or χ² against any run parameter, optionally minus another one
  (e.g. `vna_power − External_attenuation`). Click a point to open that run. Export everything as CSV.

Fits marked ⚠ are suspicious: Qi ≤ 0, a relative error > 100 %, or f_r outside the window. The usual
cause is a resonance driven into the nonlinear regime (high power) or a window that is too narrow/wide.
They are hidden in the batch plot by default but kept in the table and CSV.

### Fit options

* **Port:** notch (side-coupled, S21; Q_i is the diameter-corrected `Qi_dia_corr`, as in RFP) or
  reflection (S11).
* **Cable delay:** fitted with an initial guess (`guessdelay=True`, RFP default), fitted without the
  guess, or fixed to a value in ns. (Reflection fits never use the guess.)
* **Refine with full-model fit** (on by default): the `resonator_tools` circle fit is used as the
  starting point of a least-squares fit of the complete model (fr, Q_L, |Q_c|, φ, delay, amplitude,
  phase) to the complex data. The circle fit determines delay, circle and phase in separate steps, so a
  small delay error leaks into Q, worst for symmetric (φ ≈ 0) resonances; the full fit removes that and
  its covariance gives error bars that match the real scatter. If the circle fit lands far off (common for
  strongly over-coupled resonances in noise) the refinement also starts from values read off the data and
  keeps the best fit. Untick it to get the plain RFP circle-fit numbers.

### Goodness of fit: `res/noise`

Each fit reports the rms residual divided by the noise level of the data (estimated from
point-to-point scatter). **≈ 1 means the model describes the measurement to within the noise.**
Values ≫ 1 mean the line shape is not a Lorentzian, typically a resonator driven into the nonlinear
(Kerr) regime; such fits are flagged ⚠ above 3. In the 16-pixel Al chip data, experiment 9 (30 dB
attenuation) gives res/noise = 1.0 for every resonance at low power, while experiment 10 (0 dB) is
already distorted at its lowest power. The batch plot can show res/noise against power.

### Auto-detect options

All defaults adapt to each trace; change them when the defaults miss something.

| option | default | meaning |
|---|---|---|
| Min depth (dB) | 6 × trace noise (≥ 0.3 dB) | minimum dip depth below the baseline |
| Window (× FWHM) | 8 | fit window half-width, clipped halfway to the neighbouring resonance |
| Baseline (pts) | automatic | running-median window; grown automatically until it is much wider than the widest dip, and replaced by a straight line through the edges for zoomed single-resonance sweeps |

## Tests

```bash
pip install pytest httpx
pytest tests/          # ~30 s
```

`tests/test_fits.py` fits synthetic resonances with known parameters, generated from the same model
(`tests/synthetic.py`), over coupling regimes from 30× over-coupled (Qc ≪ Qi) to 10× under-coupled (Qi ≪ Qc), asymmetric (φ ≠ 0),
low and high Q, notch and reflection, with and without cable delay. It checks that noise-free data is
recovered exactly, that noisy fits are unbiased with error bars matching the scatter (pull std ≈ 1),
that a Kerr-distorted dip is flagged, that auto-detect finds every resonance of a multi-resonance trace
with baseline ripple, and the full path QCoDeS database → web API → fit results.
`python tests/plot_synthetic.py` draws the test resonances with their fits (`tests/synthetic_fits.png`).

## Other databases

Nothing is tied to parameter names. For each run the parameters are classified from the QCoDeS
run description (`depends_on`, label, unit):

* **frequency**: the setpoint whose name/label/unit looks like a frequency (Hz)
* **magnitude**: names/labels containing mag, amp, abs, S21, … (dB if the unit says dB, otherwise linear)
* **phase**: phase, angle, arg (deg or rad, from the unit or from the value range); always unwrapped
* **I/Q**: `I`/`Q`, `*_re`/`*_im`, real/imag, or a complex parameter → converted to magnitude/phase
* **other setpoints** (e.g. power in a 2D sweep) → one trace per value, picked in the *Trace* selector
* **scalars** (power, temperature, bandwidth, attenuation, …): anything that depends on nothing;
  shown in the sidebar and available as batch X variables

Both array storage (one blob per sweep, as `vna_*` drivers write) and point-per-row storage
(`do1d`/`do2d`-style) are supported. If a run's columns are classified wrongly, the *columns* line
in the sidebar shows which ones were picked; adjust the regexes at the top of `data.py`.

## Files

| file | |
|---|---|
| `app.py` | FastAPI server + CLI |
| `data.py` | read-only QCoDeS SQLite access and parameter classification |
| `fitting.py` | resonance detection and `resonator_tools` fits |
| `resonator_tools/` | RFP-Software's fitting library (patched: NumPy compatibility, robust cable-delay search, fixed-delay and reflection-fit bugs) |
| `tests/` | synthetic-resonance tests (`pytest tests/`) |
| `static/` | the web page (Plotly.js is vendored, so it works offline) |
