#!/bin/bash
set -e

echo "=== [1] Updating system ==="
sudo apt update -y
sudo apt install -y ca-certificates curl gnupg python3 python3-pip python3-venv

echo "=== [2] Installing Docker (official repo) ==="
sudo install -m 0755 -d /etc/apt/keyrings

if [ ! -f /etc/apt/keyrings/docker.asc ]; then
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc
fi

# Add Docker apt repo if not exists
if [ ! -f /etc/apt/sources.list.d/docker.sources ]; then
    source /etc/os-release
    sudo tee /etc/apt/sources.list.d/docker.sources > /dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: ${UBUNTU_CODENAME:-$VERSION_CODENAME}
Components: stable
Signed-By: /etc/apt/keyrings/docker.asc
EOF
fi

sudo apt update -y
sudo apt install -y docker-ce docker-ce-cli containerd.io \
                    docker-buildx-plugin docker-compose-plugin

echo "=== [3] Starting Docker ==="
sudo systemctl start docker

echo "=== [4] Creating Python virtual environment ==="
python3 -m venv venv
source venv/bin/activate

