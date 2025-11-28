
#!/bin/bash
# build.sh - Build the Docker image with your user ID

#   --no-cache \
docker build \
  --progress=plain \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  --build-arg USERNAME=$(whoami) \
  -t mtbench:latest \
  -f Dockerfile .

echo "Docker image built successfully!"
echo "Run with: ./run.sh"
