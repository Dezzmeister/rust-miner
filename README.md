# TsengToken Miner (WIP)

Mining software for TsengToken. Separate from wallet software which will be written in Go.

## Development

To develop the CPU code, clone this repo and build with

`cargo build`

or

`cargo build --release`.

To develop the GPU code as well, clone [rust-miner-kernel](https://github.com/Dezzmeister/rust-miner-kernel.git) next to this repo. To build the GPU code as well, run

`cargo build [--release] --features kernels`

Note that this will build the GPU + CPU code. You can still do `cargo build [--release]` to build just the CPU code.

## Architecture

"Mining" a TsengToken block consists of guessing a random number, appending it to the block, and hashing the block with `sha256` such that the hash is less than a certain number determined by the difficulty of the network. The goals are to maximize the amount of random numbers that can be guessed at one time and to minimize the amount of time it takes to guess a number + hash the block + check if it's good.

We can leverage the GPU to achieve the first goal. [rust-miner-kernel](https://github.com/Dezzmeister/rust-miner-kernel.git) contains a CUDA kernel that does part of the hashing. The size of a TsengToken block is such that when it is hashed with `sha256`, it will be split into 7 chunks each of 512 bits. The random guess is appended to the block, so the first five chunks will be the same regardless of this guess and these chunks can be hashed by the CPU. The GPU picks up where the CPU left off by guessing random numbers and hashing the last two chunks.
