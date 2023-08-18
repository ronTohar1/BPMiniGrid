from configuration import Config
import subprocess
from configuration import config_list
import argparse

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
            r = subprocess.run(['sbatch' , '--qos=achiya --partition=rtx2080','train_doorkey.sh', run_string])
        else:
            r = subprocess.run(['sbatch','train_doorkey.sh', run_string])
    config_string = config.to_string()
    print("Slow:\n" if slow else "Fast:\n", config_string)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slow",'-s', action="store_true", help="Run the jobs on the slow queue")

    args = parser.parse_args()  

    slow = args.slow
    for config in config_list:
        run_config(config, slow=slow)

if __name__ == "__main__":
    main()