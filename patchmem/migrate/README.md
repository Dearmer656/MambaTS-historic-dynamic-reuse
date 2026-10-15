# Running PAT-304 off the cluster (during the Ubuntu 24.04 maintenance window)

Target: a local 4080 (24GB) or a lab server GPU. `dataset/` and `checkpoints/` are gitignored on the
cluster too, so they travel separately from the code.

## 1. Code
```bash
git clone -b dearmer656/pat-304-mambats-memory git@github.com:Dearmer656/MambaTS-historic-dynamic-reuse.git
cd MambaTS-historic-dynamic-reuse
```
This is the exact commit the Linear predeg/review comments refer to (b81ad4e and later). If SSH isn't set
up on the new machine, use the https remote instead:
`https://github.com/Dearmer656/MambaTS-historic-dynamic-reuse.git`.

## 2. Environment
```bash
conda create -n mamba_iso python=3.12 -y && conda activate mamba_iso
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r patchmem/migrate/requirements-lock.txt
```
See the comments in `requirements-lock.txt` if causal-conv1d/mamba-ssm need to build from source.

## 3. Verify before touching data (no dataset needed -- these use synthetic tensors + one real Weather
   window read straight off disk for the causality check)
```bash
python patchmem/test_mem_retrieval.py   # must print ALL PASS
python patchmem/test_step_sg.py
python patchmem/test_from_scratch.py
```
If any of these fail on the new machine but passed on the cluster, it's an environment mismatch, not a
code bug -- check `torch.__version__`, `torch.version.cuda`, and `nvidia-smi` first.

## 4. Data (only ETTh2.csv + weather.csv needed for the queued runs; no checkpoint needed, everything
   queued right now trains from scratch, A0C_CKPT unset)
```bash
mkdir -p dataset/ETT-small dataset/weather
scp <cluster-login>:/project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch/dataset/ETT-small/ETTh2.csv dataset/ETT-small/
scp <cluster-login>:/project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch/dataset/weather/weather.csv dataset/weather/
```
(8MB total.) Only pull `checkpoints/` (58MB) if later running the frozen-A0 diagnostics
(diag_carry_decomp etc.) that load an `A0C_CKPT`.

## 5. Running what the sbatch scripts would have run, directly
The sbatch files under `patchmem/slurm/` are themselves the exact command lines (minus the `#SBATCH`
header and the `cd`/`export PYTHONPATH` lines, which don't apply here). E.g. for the smoke test the
`sbatch` body is just `python -u patchmem/test_mem_retrieval.py`, already covered in step 3. For a real
training run, open the relevant `*.sbatch` file, skip everything before the final `python -u ...` line,
and run that directly with the env vars it expects, e.g.:
```bash
A0C_CORR=gated A0C_MODE=cross A0C_STEP=0.1 A0C_SCRATCH_LR=0.3 A0C_STEP_MEM=0.1 A0C_MEM=real \
  python -u patchmem/run_a0corr.py --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/weather --data_path weather.csv --enc_in 21 --dec_in 21 --c_out 21 \
  --model_id weather_720_96 --model MambaTS --data custom --features M --seed 3047 \
  --seq_len 720 --label_len 48 --pred_len 96 --e_layers 3 --d_layers 2 --factor 1 \
  --des ExpMemM1Ws3047 --itr 1 --n_heads 16 --d_model 32 --d_ff 512 --dropout 0.2 \
  --patch_len 48 --stride 48 --VPT_mode 1 --ATSP_solver SA --train_epochs 10 --patience 3 \
  --batch_size 8 --learning_rate 0.0005
```
A single Weather run took ~4000s/epoch on a 3090 (`patchmem/logs/620147_a0cw.txt`); a 4080 should be
similar or faster. Nothing here is SLURM-specific -- `run_a0corr.py` / `run_mem.py` are plain scripts.

## 6. When the cluster is back
Results from off-cluster runs are not directly comparable to prior numbers unless the GPU/driver/library
versions match closely enough -- re-run at least one seed on the cluster to confirm before reporting a
verdict on Linear.
