# Docker

## Building the docker container

_NOTE: This step is not necessary if you simply want to use an already published image to run the example code on the UA HPC._

```
docker build -f Dockerfile -t uazhlt/pytorch-example .
```

## Verify PyTorch version

```
docker run --rm -it uazhlt/pytorch-example python -c "import torch; print(torch.__version__)"
```

## Publish to DockerHub

_NOTE: This step is not necessary if you simply want to use an already published image to run the example code on the UA HPC._

```
# login to dockerhub registry
docker login --username=yourdockerhubusername --email=youremail@domain.com

docker push org/image-name:taghere
```

# Singularity

## Building a Singularity image

Building a Singularity image from a def file requires sudo on a Linux system.  In this tutorial, we avoid discussing details on installing Singularity.  If you're feeling adventurous, take a look at [the example def file in this repository](./Singularity) and the official documentation:

- https://sylabs.io/guides/3.0/user-guide/installation.html


### Alternatives

#### Cloud builds
- GitHub actions:
    - [Example GitHub Workflow](https://github.com/singularityhub/github-ci/blob/master/.github/workflows/go.yml)
    - [GitHub-hosted runners](https://help.github.com/en/actions/automating-your-workflow-with-github-actions/virtual-environments-for-github-hosted-runners#supported-runners-and-hardware-resources)

#### VMs
- [Vagrant box](https://sylabs.io/guides/3.0/user-guide/installation.html#singularity-vagrant-box)

#### Docker -> Singularity
- [`docker2singularity`](https://github.com/singularityhub/docker2singularity)



## Retrieving a published Singularity image

Instead of building from scratch, we'll focus on a shortcut that simply wraps docker images published to DockerHub.

```
singularity pull uazhlt-pytorch-example.sif docker://uazhlt/pytorch-example:latest
```

# HPC

If you intend to test out [the PyTorch example included here](./example), you'll want to clone this repository:

```bash
git clone https://github.com/ua-hlt-program/pytorch-example.git
```

## Running Singularity in an interactive PBS job

Next, we'll request an interactive job (tested on El Gato):

```bash
qsub -I \
-N interactive-gpu \
-W group_list=mygroupnamehere \
-q standard \
-l select=1:ncpus=2:mem=16gb:ngpus=1 \
-l cput=3:0:0 \
-l walltime=1:0:0
```

_NOTE: If you're unfamiliar with `qsub` and the many options in the command above seem puzzling, you can find answers by checking out the manual via `man qsub` _

If the cluster isn't too busy, you should soon see a new prompt formatted something like `[netid@gpu\d\d ~]`.  

Now we'll run the singularity image we grabbed earlier.  Before that, though, let's ensure we're using the correct version of Singularity and that the correct CUDA version is available to Singularity:

```
module load singularity/3.2.1
module load cuda10/10.1
```

Now we're finally ready to run the container:

```
singularity shell --nv --no-home /path/to/your/uazhlt-pytorch-example.sif
```

If you ran into an error, check to see if you replaced `/path/to/your/` with the correct path to `uazhlt-pytorch-example.sif` before executing the command.

We're now in our Singularity container! If everything went well, we should be able to see the gpu:

```
nvidia-smi
```

You should see output like the following:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.87.01    Driver Version: 418.87.01    CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla K20Xm         On   | 00000000:8B:00.0 Off |                    0 |
| N/A   17C    P8    18W / 235W |      0MiB /  5700MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

Success (I hope)!  Now let's try running PyTorch on the GPU with batching...

# PyTorch example

The Pytorch example code can be found under [`example`](./example).  The data used in this example comes from from Delip Rao and Brian MacMahan's _Natural Language Processing with PyTorch_:

- https://github.com/joosthub/PyTorchNLPBook/tree/master/data#surnames

The dataset relates surnames to nationalities.  Our version (minor modifications) is nested under [examples/data](./examples/data).

`train.py` houses a command line program for training a classifier.  The following invocation will display the tool's help text:

```
python train.py --help
```

The simple model architecture operates is based on that of deep averaging networks (DANs; see https://aclweb.org/anthology/P15-1162/).

Reading through train.py you can quickly see how the code is organized.  Some parts (ex. `torchtext` data loaders) may be unfamiliar to you.

# Next steps

Now that you've managed to run some example PyTorch code, there are many paths forward:

- Experiment with using pretrained subword embeddings (both fixed and trainable).  Do you notice any improvements in performance/faster convergence?
- Try improving or replacing the naive model defined under `models.py`.
- Add an evaluation script for a trained model that reports macro P, R, and F1.  Feel free to use `scikit-learn`'s classification report.
- Add an inference script to classify new examples.
- Monitor validation loss to and stop training if you begin to overfit.
- Adapt the interactive PBS task outlined above to a PBS script that you can submit to the HPC.
- Address the class imbalance in the data through downsampling, class weighting, or another technique of your choosing.