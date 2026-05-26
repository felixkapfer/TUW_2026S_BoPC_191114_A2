# BoPC Assignment 2 - Option 2 CUDA

## Structure

```text
.
├── main.py                  # root entry point for helper commands
├── src/                     # small Python helper CLI
│   ├── cli.py
│   ├── helpers/
│   │   ├── commands.py
│   │   └── paths.py
│   └── utils/
│       └── process.py
├── julia_cuda/
│   ├── src/                 # CUDA/CPU Julia set implementation
│   │   ├── kernel_gpu.cu
│   │   ├── kernel_cpu.cpp
│   │   ├── juliaset.cpp
│   │   ├── Makefile
│   │   └── Makefile.gpu
│   └── jobs/                # benchmark job scripts
└── docs/                    # assignment notes
```

## Helper Commands

```bash
python main.py check-env

python main.py --dry-run build-gpu
python main.py --dry-run build-cpu

python main.py build-gpu
python main.py build-cpu

python main.py --dry-run run-gpu --size 1000 --nrep 5
python main.py --dry-run run-gpu --size 20000 --nrep 5 --block-x 32 --block-y 1 --output task.csv
python main.py --dry-run run-cpu --size 1000 --nrep 5 --output task.csv

python main.py run-gpu --size 1000 --nrep 5
python main.py run-gpu --size 20000 --nrep 5 --block-x 32 --block-y 1 --output task.csv
python main.py run-cpu --size 1000 --nrep 5 --output task.csv
```

## Direct Commands

```bash
cd julia_cuda/src

make -B -f Makefile.gpu
make -B

./juliaset_gpu -r 1000 1000 -n 5
./juliaset_gpu -r 20000 20000 -b 32 1 -n 5 -o task.csv
./juliaset_cpu -r 1000 1000 -n 5 -o task.csv
```

## Job Scripts

```bash
cd julia_cuda/jobs

sbatch gpu_code_t2.job
sbatch gpu_code_t3.job
sbatch gpu_code_t4.job
sbatch cpu_code_t4.job
```
