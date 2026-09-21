# Visualisations

`floodwater_showcase.gif` is the animated README overview. Its six video panels sample frames at regular temporal intervals—two sources each from the train, validation, and test splits—to show water-mask propagation as the scenes change. The bottom row cycles through telemetry-bearing manual evaluation examples for comparison. Water masks are shown in blue.

`floodwater_showcase.jpg` is a static version containing the same six video sources and three manual evaluation examples.

Both files are reproducible when the video corpus and manual evaluation set are extracted to `data/floodwater_dataset` and `data/floodwater_manual`:

```bash
python -m visualisations.create_showcase
```

Alternative locations are selected through `FLOODWATER_DATA`, `FLOODWATER_MANUAL_DATA`, `--data-root`, and `--manual-data-root`.
