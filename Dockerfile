FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH="/opt/conda/bin:${PATH}"

# Create a user that matches the host user (will be overridden by build args)
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=user

RUN groupadd -g ${GROUP_ID} ${USERNAME} || true && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USERNAME}

RUN chown -R ${USER_ID}:${GROUP_ID} /opt/conda


# Set working directory
WORKDIR /workspace

# Switch to the user
USER ${USERNAME}

# Initialize conda for the user
RUN conda init bash
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Copy only the environment file for dependency installation
# This will be mounted as read-only, but we copy it to cache the layer
COPY --chown=${USER_ID}:${GROUP_ID} env.yaml /tmp/env.yaml
RUN which conda
# Create conda environment
RUN conda env create --prefix /opt/conda/envs/mtbench2 -f /tmp/env.yaml

RUN conda env list

# Make RUN commands use the conda environment
SHELL ["conda", "run", "-n", "mtbench2", "/bin/bash", "-c"]

# Set up Isaac Gym (requires the isaacgym directory to be present)
# This will be installed at runtime via the entrypoint script if not already done

# Set environment variables for NVIDIA GPU
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

RUN pip install numpy==1.19

# Create an entrypoint script that will install dependencies at first run
USER root
USER root
COPY <<'EOF' /entrypoint.sh
#!/bin/bash
set -e

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate mtbench2

# Check if Isaac Gym is installed
if [ ! -f /workspace/.isaac_gym_installed ]; then
    echo "Installing Isaac Gym..."
    if [ -d "/workspace/isaacgym/python" ]; then
        cd /workspace/isaacgym/python
        pip install -e .
        cd /workspace
    else
        echo "Warning: Isaac Gym not found at /workspace/isaacgym"
    fi
fi

# Check if main package is installed
if [ ! -f /workspace/.main_package_installed ]; then
    echo "Installing main package..."
    if [ -f "/workspace/setup.py" ]; then
        cd /workspace
        pip install -e .
    fi
fi

# Execute the command
exec "$@"
EOF

RUN ls /opt/conda/envs
RUN chmod +x /entrypoint.sh

USER ${USERNAME}

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]