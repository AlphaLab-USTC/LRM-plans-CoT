# LRM-plans-CoT

The implementation of the "On Reasoning Strength Planning in Large Reasoning Models"

## Usage

Generate the main results (i.e., linear regression, direction vector extraction and analysis) in this paper:

```bash
bash scripts/analyze.sh
```

Evaluate the effect of steering:

```bash
python AnalyzeSteerFull.py
```

Replicate the results of the overthink detection before generation:

```bash
python eval_overthink.py
```

Replicate the results of the efficient inference:

```bash
python efficient_reasoning.py
```


