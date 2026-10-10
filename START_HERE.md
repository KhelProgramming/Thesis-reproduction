# Start the AgriEye baseline reproduction

The project implements the manuscript's documented **50-epoch** recipe. It does not yet establish reproduction of the final **100-epoch, five-class** model. Your current dataset has four classes.

1. Extract the project ZIP. Open a terminal inside `agrieye_reproduction`.
2. Reuse your working GPU PyTorch through a separate environment:

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py doctor
```

3. Prepare the full extracted dataset. Replace the quoted path with the folder containing `data.yaml`, `train`, `valid`, and `test`:

```bash
python run.py prepare --source "/absolute/path/to/AgriEye_export" --name agrieye_current
```

4. Review the audit output, then run the small diagnostic check:

```bash
python run.py smoke --dataset agrieye_current
python run.py bundle-records
```

Return the `records_to_share_....zip` printed by the last command. It contains reports and logs, not your dataset images or trained weights.

The smoke check uses at most 32 training and 16 validation images, one epoch, and batch 1. It checks the pipeline rather than testing the thesis hypothesis.

After those results are understood, the documented baseline command is:

```bash
python run.py train --dataset agrieye_current --recipe configs/disease_paper_50e.yaml --seed 0
python run.py benchmark
```

That full recipe requests batch 16 and may require more than the GTX 1650's 4 GB VRAM. A smaller batch must be an explicit, recorded adaptation. The Colab launcher in `notebooks/` uses these same scripts and can use a larger GPU if available.

Ask the authors for the dataset containing Healthy, the final `args.yaml` and `results.csv`, the starting/checkpoint history, and which split and box/mask metrics produced Table 12. Meanwhile, the current four-class data can be audited and used for a pipeline/recipe pilot.

The full guide, source notes, extracted paper code, and verification record are included. No new illumination transforms have been added.
