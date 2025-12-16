# CMPE214 Project 

## Prerequisites
This project uses [uv](https://docs.astral.sh/uv/) to manage all the prerequisites. To generate the enviroment, just install uv and then in this directory use:

```
uv sync
```

to generate the virtual enviroment for all the experiments.

## Generating results.

We wrote the `test.py` script to run all the tests. The script runs the experiment for all the prefetch factors for one batch size and stores the results in the csv file indicated by the `--results-file` option, an example command to run the script looks like this:

```
srun python test.py --num-sequences 1000000 --batch-size 32 --no-tqdm-enabled --backend gloo --results-file final_results.csv
```

The script automatically detects if the experiments are being run on multiple nodes from the SLURM enviroment variables, but still needs the `MASTER_ADDR` enviroment variable to set to be able to communicate with the other nodes.

We ran all the tests by running the `run_multiple_nodes.sh` batch script multiple times. One time for each node configuration (1-8 nodes) which can be changed by changing the `#SBATCH --nodes=<num_of_nodes>` SLURM directive in the script. The script can be launched on the CoE HPC with this command:

```
sbatch run_multiple_nodes.sh
```

## Geneating the plots

All the plots used in the report and the presentation can be generated with the `plot.py` script, it expects the results in the `final_results.csv` file stored in this directory. It stores all the plots in the `Plots/` directory.
