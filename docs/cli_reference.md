# CLI Reference

After installing the package in editable mode:

```bash
pip install -e ".[dev,plot]"
```

the command `pi-classifier` becomes available.

## Predict

```bash
pi-classifier predict --material Copper --soil "Black Sand / Magnetite Soil" --depth 0.65
```

Optional parameters:

```bash
pi-classifier predict \
  --material Steel \
  --soil "Wet Clay Soil" \
  --depth 0.90 \
  --distance 0.30 \
  --samples-per-material 700 \
  --noise-samples 700
```

## JSON Output

```bash
pi-classifier predict --material Copper --depth 0.65 --json
```

## Evaluate

```bash
pi-classifier evaluate
```

## Evaluate as JSON

```bash
pi-classifier evaluate --json
```
