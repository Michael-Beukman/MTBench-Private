#!/bin/bash
WANDB_API_KEY=$(cat ./wandb_key)

interactive=false
join_gpus=false

while getopts :htj opt; do
    case $opt in 
        h) show_some_help; exit ;;
        t) interactive=true ;;
        j) join_gpus=true ;;
        :) echo "Missing argument for option -$OPTARG"; exit 1;;
       \?) echo "Unknown option -$OPTARG"; exit 1;;
    esac
done

# here's the key part: remove the parsed options from the positional params
shift $(( OPTIND - 1 ))

# Check if script arguments are provided after GPU arguments
if [ $# -lt 1 ]; then
    echo "Usage: ./run.sh [options] <gpu_ids...> [-- <script_and_args>]"
    echo "Options:"
    echo "  -h: Show help"
    echo "  -t: Run in interactive mode"
    echo "  -j: Join GPUs (run single container with all specified GPUs)"
    exit 1
fi

# Find the position of -- if it exists
script_start_pos=-1
for ((i=1; i<=$#; i++)); do
    if [ "${!i}" == "--" ]; then
        script_start_pos=$i
        break
    fi
done

# Extract GPU IDs and script arguments
if [ $script_start_pos -ne -1 ]; then
    # Get GPUs (all arguments before --)
    gpus=("${@:1:$((script_start_pos-1))}")
    # Get script and args (all arguments after --)
    script_and_args="${@:$((script_start_pos+1))}"
else
    # No -- found, assume all arguments are GPU IDs
    gpus=("$@")
    script_and_args=""
fi

if [ "${gpus[0]}" == "all" ]; then
    gpus=(0 1 2 3 4 5 6 7)
fi

if [ $interactive == true ]; then
    script_and_args="bash"
    args="-it"
else
    args="-d -t"
fi

host=`hostname`

# Function to run the docker container with given GPU specification
run_container() {
    local gpu_spec="$1"
    local container_name="$2"
    
    docker run \
        --gpus "$gpu_spec" \
        -e WANDB_API_KEY=$WANDB_API_KEY \
        -e PYTHONPATH=/workspace \
        -v $(pwd):/workspace \
        --name "$container_name" \
        --rm \
        --shm-size 80G \
        $args mbeukman/michael:mtbench \
        /bin/bash -c "export LD_LIBRARY_PATH=$(conda info --base)/envs/mtbench2/lib:$LD_LIBRARY_PATH; $script_and_args"
}

if [ $join_gpus == true ]; then
    # Create a single container with all GPUs
    gpu_list=$(printf ",%s" "${gpus[@]}")
    gpu_list=${gpu_list:1}  # Remove leading comma
    container_name="mbeukman_sapg_multi_$(echo $gpu_list | tr ',' '_')"
    
    echo "Launching container $container_name on GPUs $gpu_list"
    run_container "\"device=$gpu_list\"" "$container_name"
else
    # Create separate containers for each GPU
    for gpu in "${gpus[@]}"; do
        name="mbeukman_sapg_$gpu"_`date -u +%Y-%m-%dT%H-%M-%SZ`
        echo "Launching container $name on GPU $gpu"
        run_container "device=$gpu" "$name"
    done
fi