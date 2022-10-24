pub mod hash;

use rand::prelude::*;
use serde::Serialize;
use cust::prelude::*;
use std::error::Error;

use crate::hash::start_hash;

type Wallet = [u8; 16];

#[derive(Serialize)]
struct Transaction {
    sender: Wallet,
    receiver: Wallet,
    amount: f32,
} // 36 bytes

#[derive(Serialize)]
struct RawBlock {
    id: u32,
    data: [Transaction; 10],
    parent: u32,
} // 368 bytes

#[derive(Serialize)]
pub struct FullBlock {
    raw_block: RawBlock,
    rand_token: [u8; 16],
} // 384 bytes

const REWARD_SENDER: Wallet = [0; 16];
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

fn main() -> Result<(), Box<dyn Error>> {
    #![allow(unused_mut)]

    let mut miner: Wallet = [0; 16];
    miner[0] = 69;

    let rand_tokens: [u8; 16] = rand::random();
    let guess = FullBlock { raw_block: make_rand_block(miner), rand_token: rand_tokens };

    let bytes = bincode::serialize(&guess)?;
    let cpu_hash = hash::hash_sha256(bytes.as_slice());

    let (schedule, hash) = start_hash(&guess);

    let _ctx = cust::quick_init();
    let module = Module::from_ptx(MINER_PTX, &[])?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;    

    let rand_tokens_gpu = DeviceBuffer::from_slice(&rand_tokens)?;
    let schedule_gpu = DeviceBuffer::from_slice(&schedule[0..12])?;
    let hash_vars_gpu = DeviceBuffer::from_slice(&hash)?;
    let mut hashes_gpu = DeviceBuffer::from_slice(&[0 as u32; 8])?;

    let hash_kernel = module.get_function("finish_hash")?;

    unsafe {
        launch!(
            hash_kernel<<<1, 1, 0, stream>>>(
                rand_tokens_gpu.as_device_ptr(),
                rand_tokens_gpu.len(),
                schedule_gpu.as_device_ptr(),
                hash_vars_gpu.as_device_ptr(),
                hashes_gpu.as_device_ptr()
            )
        )?;
    }

    let mut gpu_hash = [0 as u32; 8];
    stream.synchronize()?;

    hashes_gpu.copy_to(&mut gpu_hash)?;

    println!("CPU hash: {:x?}", cpu_hash);
    println!("GPU hash: {:x?}", gpu_hash);

    Ok(())
}
