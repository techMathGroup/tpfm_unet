# TPFM Unet


## Running an Experiment

To run an experiment using the TPFM Unet model using the default configuration, execute the following command in your terminal:

```bash
python src/train.py
```

To run a specific experiment configuration, use the `--config-path` argument followed by `--config-name` and the configuration file name. For example:

```bash
python src/train.py --config-path ../configs --config-name mixer_128_upscaled_x2
```

## Running MLflow

To run MLflow for tracking experiments, you can use the following command in your terminal:

```bash
mlflow ui -p 1234
```

This will start the MLflow UI on port 1234. You can then access it by navigating to `http://localhost:1234` in your web
browser.