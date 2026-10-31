# Method figure parts

Final method, q_s 1/4, schedule B, cross_hr (entropy space at the global threshold +
readout-attention time). Runs: `exp.viz.paper.method_scan.RUNS` (s16m8, s8m4).

    python -m exp.viz.paper.method_scan --setup s8m4                 # scan_<setup>.pt, scan_top_<setup>.png
    python -m exp.viz.paper.method_parts --setup s8m4 --tiles 30093@r0c0 40220@r0c64 30201@r64c0
    cd figures/paper/method && python ../../../exp/viz/paper/method_preview.py s8m4 preview_s8m4.png <tiles>

`<setup>/<tile>/stack_*.png`: transparent diagonal stacks, first time step in front, a fixed
60 px step between frames; unpicked windows transparent. `single/` = every slice; `parts.json`
= picks. S2 shown as MEDIAN composites of the raw dates (nodata dropped), display only.
Scan rankings skip tiles whose S2 mean composites contain -10000 nodata frames.
