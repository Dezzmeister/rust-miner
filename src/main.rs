pub mod hash;

use rand::prelude::*;
use serde::Serialize;
use cust::{prelude::*, device::DeviceAttribute};
use std::env;
use std::error::Error;
use std::time::{Instant, Duration};

use crate::hash::{start_hash, hash_block};

type Wallet = [u8; 16];

#[derive(Serialize, Debug, Clone, Copy)]
struct Transaction {
    sender: Wallet,
    receiver: Wallet,
    amount: f32,
} // 36 bytes

#[derive(Serialize, Debug, Clone, Copy)]
struct RawBlock {
    id: u32,
    data: [Transaction; 10],
    parent: u32,
} // 368 bytes

#[derive(Serialize, Debug)]
pub struct FullBlock {
    raw_block: RawBlock,
    rand_token: [u8; 16],
} // 384 bytes

pub struct Winner {
    rand_token: [u8; 16],
    hash: [u32; 8],
}

pub struct MiningResult {
    winner: Winner,
    tries: u64,
}

trait ToHexString {
    fn to_hex_string(&self) -> String;
}

impl ToHexString for u8 {
    fn to_hex_string(&self) -> String {
        format!("{:02x}", self)
    }
}

impl ToHexString for u32 {
    fn to_hex_string(&self) -> String {
        format!("{:08x}", self)
    }
}

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

fn make_rand_guesses(num_guesses: usize) -> Vec<u8> {
    let mut out = vec![0 as u8; num_guesses * 16];
    
    for i in 0..(num_guesses * 16) {
        out[i] = rand::random();
    }

    out
}

fn stringify_join<T: ToHexString>(items: &[T]) -> String {
    let mut out = String::with_capacity(items.len());
    for item in items {
        out.push_str(&item.to_hex_string());
    }

    out
}

fn hash_from_difficulty(difficulty: u8) -> [u32; 8] {
    let mut out = [0xFFFF_FFFF as u32; 8];
    let mut tmp_diff = difficulty as usize;
    let mut idx = 0;

    while tmp_diff >= 32 {
        out[idx] = 0;
        tmp_diff = tmp_diff - 32;
        idx = idx + 1;
    }

    let mut mask: u32 = 0;
    
    while tmp_diff != 0 {
        mask = mask | (1 << (32 - tmp_diff));
        tmp_diff = tmp_diff - 1;
    }

    out[idx] = out[idx] & !mask;

    out
}

/// Compare two numbers represented as arrays of orderable "digits".
/// Each element of the arrays can be treated as a separate digit.
/// The most significant digits are first. We also assume that
/// a and b have the same size because this function needs to run
/// as quickly as possible.
fn less_than(a: &[u32; 8], b: &[u32; 8]) -> bool {
    for i in 0..a.len() {
        if a[i] == b[i] {
            continue;
        }

        return a[i] < b[i]
    }

    false
}

fn find_winner(rand_tokens: &[u8], hashes: &[u32], max_hash: &[u32; 8]) -> Option<Winner> {
    for i in (0..hashes.len()).step_by(8) {
        let mut hash = [0 as u32; 8];
        hash.copy_from_slice(&hashes[i..(i + 8)]);

        if less_than(&hash, &max_hash) {
            let token_index = i * 2;
            let mut rand_token = [0 as u8; 16];
            rand_token.copy_from_slice(&rand_tokens[token_index..(token_index + 16)]);

            return Some(Winner { rand_token, hash });
        }
    }

    None
}

fn update_random_tokens(tokens: &mut Vec<u8>) {
    for i in 0..tokens.len() {
        tokens[i] = rand::random();
    }
}

fn mine_block(block: FullBlock, winning_hash: [u32; 8]) -> MiningResult {
    #![allow(unused_mut)]

    let block_bytes = bincode::serialize(&block).expect("Failed to serialize block");

    println!("Serialized block to be mined: {}", stringify_join(&block_bytes));

    // This gets created again when we start hashing but we don't use it directly so
    // we don't need it anymore
    drop(block_bytes);

    let _ctx = cust::quick_init().expect("Failed to create CUDA context");

    let device = Device::get_device(0).expect("Failed to get CUDA device");
    println!("Device Name: {}", device.name().expect("Failed to get device name"));
    println!("Max threads per device block: {}", device.get_attribute(DeviceAttribute::MaxThreadsPerBlock).expect("Failed to get max threads per block"));

    let module = Module::from_ptx(MINER_PTX, &[]).expect("Failed to load hashing module");
    let kernel = module.get_function("finish_hash").expect("Failed to load hashing kernel");
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).expect("Failed to create stream to submit work to device");

    let (schedule, hash) = start_hash(&block);
    let (grid_size, block_size) = kernel.suggested_launch_configuration(0, 0.into()).expect("Unable to determine launch config");
    let num_guesses: usize = (grid_size * block_size).try_into().unwrap();

    println!("Running hash kernel with grid size {} and block size {}", grid_size, block_size);
    println!("Hashing {} guesses per round", num_guesses);

    let mut guesses = make_rand_guesses(num_guesses);
    let mut hashes = vec![0 as u32; num_guesses * 8];

    let mut rand_tokens_gpu = DeviceBuffer::from_slice(&guesses).expect("Failed to create device memory for random tokens");
    let mut schedule_gpu = DeviceBuffer::from_slice(&schedule[0..12]).expect("Failed to create device memory for schedule");
    let mut hash_vars_gpu = DeviceBuffer::from_slice(&hash).expect("Failed to create device memory for hash variables");
    let mut hashes_gpu = DeviceBuffer::from_slice(hashes.as_slice()).expect("Failed to create device memory for hashes");

    let mut hashes_out = vec![0 as u32; num_guesses * 8];

    let mut out: Option<Winner> = None;
    let mut tries: u64 = 0;
    let mut num_hashes: u64 = 0;
    let mut start = Instant::now();

    while out.is_none() {
        unsafe {
            launch!(
                kernel<<<grid_size, block_size, 0, stream>>>(
                    rand_tokens_gpu.as_device_ptr(),
                    rand_tokens_gpu.len(),
                    schedule_gpu.as_device_ptr(),
                    hash_vars_gpu.as_device_ptr(),
                    hashes_gpu.as_device_ptr()
                )
            ).expect("Failed to launch hashing kernel");
        }

        stream.synchronize().expect("Failed to synchronize device stream");

        hashes_gpu.copy_to(&mut hashes_out).expect("Failed to copy hashes from device back to host");
        out = find_winner(&guesses, &hashes_out, &winning_hash);

        tries = tries + 1;
        num_hashes = num_hashes + (hashes_gpu.len() as u64);

        if start.elapsed() > Duration::from_secs(5) {
            let time_elapsed = start.elapsed().as_millis();
            let hashes_per_millis = (num_hashes as u128) / time_elapsed;

            println!("Hashes per second: {}", hashes_per_millis * 1000);

            start = Instant::now();
            num_hashes = 0;
        }

        update_random_tokens(&mut guesses);
        rand_tokens_gpu.copy_from(&guesses).expect("Failed to copy random tokens to device");
    }

    MiningResult { winner: out.unwrap(), tries }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let difficulty: u8 = args[1].parse().expect("Failed to parse difficulty. Difficulty should be a small positive integer");
    let winning_hash = hash_from_difficulty(difficulty);

    println!("Difficulty is {}", difficulty);
    println!("Hash must be less than {}", stringify_join(&winning_hash));

    let mut miner: Wallet = [0; 16];
    miner[0] = 16;

    let block = FullBlock { raw_block: make_rand_block(miner), rand_token: [0 as u8; 16] };
    let raw_block = block.raw_block;
    
    let handle = std::thread::spawn(move || {
        mine_block(block, winning_hash)
    });

    let MiningResult{winner, tries} = handle.join().expect("Failed to load mining result");
    println!("Won the block in {} tries with random guess {}", tries, stringify_join(&winner.rand_token));
    println!("Block hash: {}", stringify_join(&winner.hash));
    let mined_block = FullBlock{raw_block, rand_token: winner.rand_token};
    let cpu_hash = hash_block(&mined_block).expect("Failed to verify block hash on CPU");
    println!("Verification by CPU: {}", &stringify_join(&cpu_hash));

    Ok(())
}
