Bootstrap:docker  
From:pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime

%labels
MAINTAINER myedibleenso

%setup
# commands to be executed on host outside container during bootstrap

%post
# commands to be executed inside container during bootstrap.
echo "Installing packages listed in requirements.txt using conda..."
while read requirement; do conda install --yes $requirement; done < requirements.txt
# download and run NIH HPC cuda for singularity installer
# NOTE: this should match
CUDA_VER=10.1
wget ftp://helix.nih.gov/CUDA/cuda4singularity
chmod 755 cuda4singularity
./cuda4singularity
rm cuda4singularity

%environment
RAWR_BASE=/code
export RAWR_BASE


%runscript
# commands to be executed when the container runs

%test
# commands to be executed within container at close of bootstrap process
python --version
python -c 'import torch; print(f"PyTorch v{torch.__version__}")'

%post  
# this section executes after bootstrapping the image.
echo "Running 'post' commands..."  
mkdir -p /output
 