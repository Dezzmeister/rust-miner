use sha2::{Sha256, Digest};
use rand::prelude::*;
use serde::Serialize;
use cust::prelude::*;
use std::error::Error;

type Wallet = [u8; 16];

#[derive(Serialize)]
struct Transaction {
    sender: Wallet,
    receiver: Wallet,
    amount: f32,
}

#[derive(Serialize)]
struct RawBlock {
    id: u32,
    data: [Transaction; 10],
    parent: u32,
}

#[derive(Serialize)]
struct FullBlock {
    raw_block: RawBlock,
    rand_token: [u8; 16],
}

const REWARD_SENDER: Wallet = [{0}; 16];
const BLOCK_REWARD: f32 = 16.0;

static MINER_PTX: &str = include_str!("../kernels/miner.ptx");

impl Default for Transaction {
    fn default() -> Self {
        let mut rng = rand::thread_rng();

        let sender: Wallet = rand::random();
        let receiver: Wallet = rand::random();
        let amount: f32 = rng.gen::<f32>() * 100.0;

        Transaction { sender, receiver, amount }
    }
}

fn make_block_reward(winner: Wallet) -> Transaction {
    Transaction { sender: REWARD_SENDER, receiver: winner, amount: BLOCK_REWARD }
}

fn make_rand_block(winner: Wallet) -> RawBlock {
    let mut data: [Transaction; 10] = Default::default();
    data[9] = make_block_reward(winner);

    RawBlock { data, id: 69, parent: 68 }
}

fn guess_full_block(raw_block: RawBlock) -> FullBlock {
    let rand_token: [u8; 16] = rand::random();

    FullBlock { raw_block, rand_token }
}

fn hash_full_block(full_block: &FullBlock) -> [u8; 32] {
    let mut hasher = Sha256::new();
    match bincode::serialize(full_block) {
        Ok(bytes) => {
            hasher.update(bytes);
            let vec = hasher.finalize();
            vec.try_into().unwrap()
        },
        Err(error) => panic!("{}", error)
    }
}

fn fill_rand_slice(slice: &mut [f32]) {
    for i in 0..slice.len() {
        slice[i] = rand::random();
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut miner: Wallet = [0; 16];
    miner[0] = 69;

    let raw_block = make_rand_block(miner);
    let guess = guess_full_block(raw_block);
    let hash = hash_full_block(&guess);

    println!("{:x?}", hash);

    // ======= CUDA code =======

    const SIZE: usize = 10000;

    // CUDA API must be initialized and a device must be chosen to make a context
    let _ctx = cust::quick_init()?;

    // Load the compiled PTX code into a module
    let module = Module::from_ptx(MINER_PTX, &[])?;
    
    // Create a stream to submit work to
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Make some slices (on the host for now)
    let mut x = [0.0f32; SIZE];
    let mut y = [0.0f32; SIZE];
    let mut out = [0.0f32; SIZE];

    fill_rand_slice(&mut x);
    fill_rand_slice(&mut y);

    // Copy the slices to device memory
    let x_gpu = x.as_slice().as_dbuf()?;
    let y_gpu = y.as_slice().as_dbuf()?;
    let out_gpu = out.as_slice().as_dbuf()?;

    // Get the kernel so we can invoke it directly
    let sum_func = module.get_function("add")?;

    // Find an optimal launch config
    let (_, block_size) = sum_func.suggested_launch_configuration(0, 0.into())?;
    let grid_size = (SIZE as u32 + block_size - 1) / block_size;

    println!("Using {} blocks and {} threads per block", grid_size, block_size);

    unsafe {
        // Launch the kernel. Slices need their lengths passed in after the device ptr
        launch!(
            sum_func<<<grid_size, block_size, 0, stream>>>(
                x_gpu.as_device_ptr(),
                x_gpu.len(),
                y_gpu.as_device_ptr(),
                y_gpu.len(),
                out_gpu.as_device_ptr()
            )
        )?;

        println!("after launch");
    }

    // Block until the work is complete
    stream.synchronize()?;
    
    // Copy result data on device back to host
    out_gpu.copy_to(&mut out)?;

    println!("{} + {} = {}", x[0], y[0], out[0]);

    Ok(())
}
