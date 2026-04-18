# sam environment

Conda environment and bootstrap scripts for [Facebook SAM](https://huggingface.co/facebook/sam-audio-base)

[GitHub](https://github.com/facebookresearch/sam-audio)

## model

You have to ask for access to the [SAM model](https://huggingface.co/facebook/sam-audio-base) on Hugging Face

Export the environment variable `HF_TOKEN` with the HF's token so the application could download this model.

## IMPORTANT

Facebook SAM model loads and processes **the whole track** at once. So the longer the track the higher chance of getting OOM.

The input file could be split into a sequence of overlapped chunks, processed independently and then mixed
together back to a single track. And overlapping would help to avoid clicks.

The problem though is that it could lead to different timbral characteristics of each chunk. Which is not
what anyone would expect from such type of processing.

If you need to process full track, probably worth to use CPU mode on the host with a lot of RAM.

## install

```shell
# first, make an isolated Conda environment with Python, Poetry and CUDA inside
$ make env-init-conda

# then install the most of the dependencies with Poetry
$ make env-init-poetry
```

## run

```shell
bin/sam.sh \
  --source "work/audio.mp3" \
  --query "extract drum track" \
  [--model-type LARGE] \
  [--device-type AUTO] \
  [--data-type FLOAT32] \
```

## model type

- [LARGE](https://huggingface.co/facebook/sam-audio-large)
- [BASE](https://huggingface.co/facebook/sam-audio-base)
- [SMALL](https://huggingface.co/facebook/sam-audio-small) (default)

## device types

- AUTO (default, CUDA or CPU, depends on GPU availability)
- CUDA
- CPU

## data type

- FLOAT32 (default)
- FLOAT16
- BFLOAT16
