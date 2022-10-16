use sha2::{Sha256, Digest};
use rand::prelude::*;
use serde::Serialize;

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

fn main() {
    let mut miner: Wallet = [0; 16];
    miner[0] = 69;

    let raw_block = make_rand_block(miner);
    let guess = guess_full_block(raw_block);
    let hash = hash_full_block(&guess);

    println!("{:x?}", hash);
}
