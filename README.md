# AiGaze

AI Gaze predictive visual attention app with Streamlit.

## Ground-Truth Evaluation (MIT/CAT style)

You can now compute validated saliency accuracy metrics against real fixation data:

- NSS
- AUC-Judd
- CC

### Expected folder layout

- `images_dir/` contains source images
- `fixation_dir/` contains fixation maps with matching filenames/stems
- optional `density_dir/` contains fixation density maps for CC with matching stems

### Run evaluator

```bash
python3 ground_truth_eval.py \
  --images_dir "/path/to/images" \
  --fixation_dir "/path/to/fixation_maps" \
  --density_dir "/path/to/density_maps" \
  --max_images 0
```

Notes:

- `--density_dir` is optional. If omitted, density maps are derived from fixation masks.
- `--max_images 0` evaluates all matched pairs.
- Output includes per-image metrics and dataset-level means as JSON.
