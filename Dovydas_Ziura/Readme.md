# MathWriting LaTeX Recognition — Final Project

This project trains a compact CNN+LSTM model to recognize handwritten math expressions and output LaTeX. It targets a modest GPU (GTX 1650 2GB) and includes a streaming dataset pipeline, scheduled sampling, preview decoding, beam search, and plotting of training curves.


## Features
- Streaming `tf.data` pipeline for big datasets (100k–200k) without huge RAM spikes
- Compact CNN encoder + 2-layer LSTM decoder, BOS/EOS tokens
- Scheduled sampling (optional) to reduce exposure bias
- Beam search with penalties and heuristics for inference
- Preview decoding during training (epoch end)
- Matplotlib training curves saved as `training_curves.png`

## Requirements
- Windows 10/11, PowerShell 5.1
- Conda environment with TensorFlow GPU working on your NVIDIA driver
- Recommended: NVIDIA Studio/Game Ready driver (updated)

### Compatibility Notes (Windows + NVIDIA + TF)
- Windows supports NVIDIA GPU acceleration with TensorFlow 2.10 (last official TF GPU build for Windows).
- Verified environment here uses Python 3.10.13 and NumPy 2.2.6 in Conda.
- cuDNN 8.1 and CUDA v11.2 tooling are compatible with TF 2.10 on Windows.

Python packages (typical):
- tensorflow (GPU build), tensorflow-addons
- numpy, pillow, datasets (Hugging Face), scikit-learn
- matplotlib

## Project Structure

## Setup
1. Create and activate your conda env, install packages:
```powershell
conda activate tf2-gpu
pip install tensorflow-addons datasets pillow matplotlib scikit-learn
```
2. Verify GPU works:
```powershell
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
Expected: a list containing `GPU:0`.

## Training
Run training with safe defaults for a 2GB GPU:
```powershell
conda activate tf2-gpu
cd "c:\Users\dovyd\OneDrive\Dokumentai\Univeras\CERN data analysis\final_project\scripts"
python .\train.py --batch-size 6 --data-limit 40000 --epochs 20 (these could also be changed in config.py)
```

Outputs:

## Prediction
Run inference on validation examples or your own image:
```powershell
conda activate tf2-gpu
cd "c:\Users\dovyd\OneDrive\Dokumentai\Univeras\CERN data analysis\final_project\scripts"
python .\predict.py (Image path could be changed in config.py)

### Batch Prediction & CSV
Predict multiple images at once and save a CSV for quick evaluation. You can provide a list, a glob pattern, or a numeric range.

Examples:
```powershell
# 1) Glob all JPGs in a folder
python .\predict.py --glob ..\dataset\hand_data\*.jpg --beam 5 --csv ..\output\predictions.csv

	- `--image <path>`: input image path
	- `--beam <N>`: beam size (1=greedy)
	- `--model <path>`: explicit model file
	- `--length-norm <x>`: length normalization exponent
	- `--output <path>`: write decoded LaTeX to a file


# 2) Explicit list
python .\predict.py --images ..\dataset\hand_data\1.jpg ..\dataset\hand_data\2.jpg ..\dataset\hand_data\7.jpg --beam 5 --csv ..\output\predictions.csv

## Tips for Large Datasets

## Demo
Quick demo assuming images named `1.jpg` to `9.jpg` exist under `dataset/hand_data`:
```powershell
conda activate tf2-gpu
cd "c:\Users\dovyd\OneDrive\Dokumentai\Univeras\CERN data analysis\final_project\scripts"
python .\predict.py --dir ..\dataset\hand_data --range-start 1 --range-end 9 --beam 5 --csv ..\output\demo_predictions.csv
```
Open the CSV in your editor and add the real LaTeX in the `ground_truth` column for quick evaluation.

## Troubleshooting

## Common Commands
```powershell
Get-Process | Where-Object {$_.ProcessName -match "^python(\.exe)?$|^py(\.exe)?$"} | Stop-Process -Force
```
```powershell
Start-Process "c:\Users\dovyd\OneDrive\Dokumentai\Univeras\CERN data analysis\final_project\scripts\training_curves.png"
```

## Notes

## LaTeX Report from Predictions

Generate a LaTeX report summarizing predictions vs ground truth and inline images.

Steps:

1. Ensure `output/predictions.csv` exists (run the prediction script first).
2. Run the generator:

```powershell
python Dovydas_Ziura\scripts\generate_latex_from_predictions.py
```

This writes `Dovydas_Ziura/output/predictions_report.tex`.

Compile to PDF (requires a LaTeX distribution like TeX Live or MiKTeX):

```powershell
pdflatex -interaction=nonstopmode -halt-on-error Dovydas_Ziura\output\predictions_report.tex
```

If `pdflatex` is not on PATH, open the `.tex` file in your LaTeX editor and compile there.