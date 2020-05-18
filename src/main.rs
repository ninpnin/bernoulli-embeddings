use std::env;
use rand::prelude::*;
extern crate rayon;
use std::time::Instant;

mod preprocess;
mod math;

// TRAINING CONSTS
const LEARNING_RATE: f32 = 0.0000025;
const PASSES: u8 = 5;
const CONTEXT_SIZE: u64 = 5;
const SYMMETRIC_CONTEXT_SIZE: u64 = CONTEXT_SIZE * 2;
const NEGATIVE_SAMPLES: u8 = 5;
const BATCH_SIZE: u64 = 10000;
const LAMBDA_0: f32 = 1.0;
const DIMENSIONALITY: u64 = 96;

// Add input's values to output
// [1,2,3], [0,1,0] -> [1,2,3], [1,3,3]
fn copy_val(input: &[f32], output: &mut[f32], dim: u64, r: f32) {
    for k in 0..dim {
        output[k as usize] = output[k as usize] + r * input[k as usize];
    }
}


fn train_embedding(data: Vec<u64>,  data_len: u64, vocab_size: u64, dimensionality: u64) {

    println!("Train embedding...");

    let embedding_len = (vocab_size * dimensionality) as usize;
    let mut word_vectors: Vec<f32> = vec![];
    let mut context_vectors: Vec<f32> = vec![];

    // Populate word vectors with random numbers
    println!("Generate word vectors...");
    for _ in 0..embedding_len {
        let p1: f32 = rand::random::<f32>() - 0.5;
        let p2: f32 = rand::random::<f32>() - 0.5;

        word_vectors.push(p1);
        context_vectors.push(p2);
    }
    println!("Done.");

    let mut word_gradient: Vec<f32> = vec![0.0; embedding_len as usize];
    let mut context_gradient: Vec<f32> = vec![0.0; embedding_len as usize];

    let start = CONTEXT_SIZE;
    let end = data_len - CONTEXT_SIZE;

    let mut rng = thread_rng();

    let mut timestamp = Instant::now();
    for data_pass in 0..PASSES {
        println!("Data pass {}", data_pass);
        for i in start..end {

            if (i + 1) % BATCH_SIZE == 0 || i + 1 == end {
                println!("Batch {} done! i: {}", i / BATCH_SIZE, i);
                println!("Duration: {} seconds", timestamp.elapsed().as_secs_f32());
                timestamp = Instant::now();
                
                // Update weights with gradients
                //let one_ratio = 1.0 / (data_len as f32);
                let batch_ratio = (data_len / BATCH_SIZE) as f32;
                for j in 0..embedding_len {

                    //let word_update = ;
                    word_gradient[j] = word_gradient[j] * batch_ratio;
                    context_gradient[j] = context_gradient[j] * batch_ratio;
                    
                    // Add priors to gradient
                    word_gradient[j] += - LAMBDA_0 * word_vectors[j];
                    context_gradient[j] += - LAMBDA_0 * context_vectors[j];

                    // Add to vectors
                    word_vectors[j] += word_gradient[j] * LEARNING_RATE;
                    context_vectors[j] += context_gradient[j] * LEARNING_RATE;
                }
                // Reset gradients
                word_gradient = vec![0.0; embedding_len as usize];
                context_gradient = vec![0.0; embedding_len as usize];
                println!("Doned.");
            }

            // Positive samples
            let word_index = data[i as usize] as usize;
            let dim = dimensionality as usize;
            let word_slice = &word_vectors[word_index..word_index+ dim];

            let mut combined_word_slice = [0.0; (DIMENSIONALITY * SYMMETRIC_CONTEXT_SIZE) as usize];
            let mut combined_context_slice = [0.0; (DIMENSIONALITY * SYMMETRIC_CONTEXT_SIZE) as usize];

            // Calculate dot product and sigma
            (1..CONTEXT_SIZE).for_each(|separation| {
                let context_ix_after = data[(i + separation) as usize] as usize;
                let context_ix_before = data[(i - separation) as usize] as usize;
                
                let context_slice_after = &context_vectors[context_ix_after..context_ix_after+dim];
                let context_slice_before = &context_vectors[context_ix_before..context_ix_before+dim];

                let start1 = (DIMENSIONALITY * separation) as usize;
                let end1 = (DIMENSIONALITY * (separation + 1)) as usize;
                let start2 = (DIMENSIONALITY * (separation + CONTEXT_SIZE)) as usize;
                let end2 = (DIMENSIONALITY * (separation + 1 + CONTEXT_SIZE)) as usize;
                (combined_context_slice[start1..end1]).copy_from_slice(context_slice_after);
                (combined_context_slice[start2..end2]).copy_from_slice(context_slice_before);

                (combined_word_slice[start1..end1]).copy_from_slice(word_slice);
                (combined_word_slice[start2..end2]).copy_from_slice(word_slice);
            });
            let dot = math::dot_prod(&combined_word_slice, &combined_context_slice, DIMENSIONALITY * SYMMETRIC_CONTEXT_SIZE);
            let sigm = math::sigmoid(-dot);

            // Update gradient
            for separation in 1..CONTEXT_SIZE {
                let context_ix_after = data[(i + separation) as usize] as usize;
                let context_ix_before = data[(i - separation) as usize] as usize;
                
                let word_slice = &word_vectors[word_index..word_index+dim];
                let context_slice_after = &context_vectors[context_ix_after..context_ix_after+dim];
                let context_slice_before = &context_vectors[context_ix_before..context_ix_before+dim];

                let mut word_gradient_slice = &mut word_gradient[word_index..word_index+dim];
                copy_val(&context_slice_after, &mut word_gradient_slice, dimensionality, sigm);
                copy_val(&context_slice_before, &mut word_gradient_slice, dimensionality, sigm);

                let mut context_gradient_slice_after = &mut context_gradient[context_ix_after..context_ix_after+dim];
                copy_val(&word_slice, &mut context_gradient_slice_after, dimensionality, sigm);
                
                let mut context_gradient_slice_before = &mut context_gradient[context_ix_before..context_ix_before+dim];
                copy_val(&word_slice, &mut context_gradient_slice_before, dimensionality, sigm);
            }

            for _ in 0..NEGATIVE_SAMPLES {
                let ns_i = rng.gen_range(start, end);
                let word_index = data[ns_i as usize];

                // Calculate dot product and sigma
                for separation in 1..CONTEXT_SIZE {
                    let context_ix_after = data[(ns_i + separation) as usize] as usize;
                    let context_ix_before = data[(ns_i - separation) as usize] as usize;
                    
                    let context_slice_after = &context_vectors[context_ix_after..context_ix_after+dim];
                    let context_slice_before = &context_vectors[context_ix_before..context_ix_before+dim];

                    let start1 = (DIMENSIONALITY * separation) as usize;
                    let end1 = (DIMENSIONALITY * (separation + 1)) as usize;
                    let start2 = (DIMENSIONALITY * (separation + CONTEXT_SIZE)) as usize;
                    let end2 = (DIMENSIONALITY * (separation + 1 + CONTEXT_SIZE)) as usize;
                    (combined_context_slice[start1..end1]).copy_from_slice(context_slice_after);
                    (combined_context_slice[start2..end2]).copy_from_slice(context_slice_before);

                }
                let dot = math::dot_prod(&combined_word_slice, &combined_context_slice, DIMENSIONALITY * SYMMETRIC_CONTEXT_SIZE);
                let ns_sigm = math::sigmoid(dot);

                // Update gradient
                for separation in 1..CONTEXT_SIZE {
                    let context_ix_after = data[(ns_i + separation) as usize] as usize;
                    let context_ix_before = data[(ns_i - separation) as usize] as usize;
                    
                    let word_index = word_index as usize;

                    let word_slice = &word_vectors[word_index..word_index+dim];
                    let context_slice_after = &context_vectors[context_ix_after..context_ix_after+dim];
                    let context_slice_before = &context_vectors[context_ix_before..context_ix_before+dim];

                    let mut word_gradient_slice = &mut word_gradient[word_index..word_index+dim];
                    copy_val(&context_slice_after, &mut word_gradient_slice, dimensionality, -ns_sigm);
                    copy_val(&context_slice_before, &mut word_gradient_slice, dimensionality, -ns_sigm);

                    let mut context_gradient_slice_after = &mut context_gradient[context_ix_after..context_ix_after+dim];
                    copy_val(&word_slice, &mut context_gradient_slice_after, dimensionality, -ns_sigm);
                    
                    let mut context_gradient_slice_before = &mut context_gradient[context_ix_before..context_ix_before+dim];
                    copy_val(&word_slice, &mut context_gradient_slice_before, dimensionality, -ns_sigm);
                }
            }
        }
    }

    println!("{:?}", &word_vectors[..100]);
    let _ = preprocess::write_embeddings(word_vectors, context_vectors);
    println!("Traineds.");
}

fn generate_random_data(data_len: u64, vocab_size: u64) -> Vec<u64> {
    println!("Generate random data...");
    let mut rng = thread_rng();
    let data = (0..data_len).map(|i| {
        if i % 100000 == 0 {
            println!("i: {}", i);
        }
        rng.gen_range(0, vocab_size)
    }).collect();
    println!("Done.");
    data
}

fn main() {
    
    let args: Vec<String> = env::args().collect();
    let dataset_name: String = args[1].to_string();
    let data = preprocess::generate_data(dataset_name);
    let data_len = data.len() as u64;
    let vocab_size = 200000;
    /*
    let data_len = 100000000;
    let vocab_size = 370000;
    let data = generate_random_data(data_len, vocab_size);
    */
    train_embedding(data, data_len, vocab_size, DIMENSIONALITY);
}
