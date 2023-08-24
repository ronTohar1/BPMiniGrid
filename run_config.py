from configuration import Config
import subprocess
from configuration import config_list
import argparse
import os

def run_config(config, slow=False, num_runs=2):


    # run_string = "--name_addition {} --agent_class {} --frame_stack {} --logdir {} --num_episodes {} --env_index {} -lr {} -na {} --features_dim {} --seed {}".format(
    #     config.name,
    #     config.agent_class,
    #     config.frame_stack,
    #     config.tensorlog,
    #     config.num_episodes,
    #     config.env_index,
    #     config.lr,
    #     config.network_architecture,
    #     config.features_dim,
    #     config.seed,
    # )

    run_string = ""
    for k,v in config.__dict__.items():
        if v is not None and v != "":
            if v is True:
                run_string += f"--{k} "
            elif v is not False:            
                run_string += f"--{k} {v} "
        
    for i in range(num_runs):
        if slow:
            os.system('sbatch --qos=achiya --partition=rtx2080 ' + 'train.sh ' + run_string)
        else:
            r = subprocess.run(['sbatch','train.sh', run_string])
    config_string = config.to_string()
    print("Slow:\n" if slow else "Fast:\n", config_string)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slow",'-s', action="store_true", help="Run the jobs on the slow queue")
    parser.add_argument("--num_runs",'-n', type=int, default=1, help="Number of runs to do for each config")

    args = parser.parse_args()  

    slow = args.slow
    num_runs = args.num_runs
    for config in config_list:
        run_config(config, slow=slow, num_runs=num_runs)

if __name__ == "__main__":
    main()