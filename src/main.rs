use std::env;
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use ndarray::{Array, Array1};
use ndarray_npy::{WriteNpyError, WriteNpyExt};
use rand::prelude::*;
use packed_simd::f32x16;
extern crate rayon;
use rayon::prelude::*;
use std::time::Instant;

const LEARNING_RATE: f32 = 0.025;
const PASSES: u8 = 1;
const CONTEXT_SIZE: u64 = 5;
const NEGATIVE_SAMPLES: u8 = 5;
const BATCH_SIZE: u64 = 100000;
const LAMBDA_0: f32 = 0.025;
const DIMENSIONALITY: u64 = 100;


fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

fn sigmoid(x: f32) -> f32 {
	let denominator = 1.0 + x.exp();
	1.0 / denominator
}

const THRESHOLD: f32 = 0.0001;
fn discard(current: u64, total: u64) -> bool {
	let ratio = (current as f64) / (total as f64);
    let p: f64 = rand::random::<f64>();
    p > ratio
}

fn clean_string(input_string: String) -> String {
    input_string
        .replace(|c: char| !c.is_alphanumeric(), "")
        .to_lowercase()
}

fn write_vector(vector: Vec<u64>) -> Result<(), WriteNpyError> {
    let arr: Array1<u64> = Array::from(vector);
    let writer = File::create("array.npy")?;
    arr.write_npy(writer)?;
    Ok(())
}

fn write_embeddings(word: Vec<f32>, context: Vec<f32>) -> Result<(), WriteNpyError> {
    let word_arr: Array1<f32> = Array::from(word);
    let context_arr: Array1<f32> = Array::from(context);
    let word_writer = File::create("word.npy")?;
    word_arr.write_npy(word_writer)?;
    let context_writer = File::create("context.npy")?;
    context_arr.write_npy(context_writer)?;
    Ok(())
}

fn generate_data(dataset_name: String) {
    println!("Fetch data from {}...", dataset_name);
	// Fetch dataset name from arguments

    let mut vocabulary: HashMap<String, u64> = HashMap::new();
    let mut vocab_size = 0;

    let mut word_count: HashMap<String, u64> = HashMap::new();

    println!("First loop...");
    
    let dir_path = format!("{}", dataset_name);
    let paths = fs::read_dir(dir_path).unwrap();
    
    let mut data: Vec<u64> = vec![];
    for path in paths {

        let path_string = path.unwrap().path();
        let contents = read_lines(path_string);

        let mut token_count = 0; 
        if let Ok(lines) = contents {
            for line in lines {
                if let Ok(clean_line) = line {
                	
                    let tokens = clean_line.split_whitespace();

                    for raw_token in tokens {
                        if token_count % 1000000 == 0 {
                            println!("Token count {}", token_count);
                        }
                        token_count += 1;

                        let token: String = clean_string(raw_token.to_string());
						let count = word_count.entry(token.clone()).or_insert(0);
    					*count += 1;

    					let vocab_index = vocabulary.entry(token).or_insert(vocab_size);
    					if *vocab_index == vocab_size {
    						vocab_size += 1;
    					}
                        
                        data.push(*vocab_index);
                    }
                }
            }
        }
    }
    println!("{:?}", vocabulary);
    println!("{:?}", word_count);

    write_vector(data);
}

// DIFFERENT IMPLEMENTATIONS OF THE DOT PRODUCT

// Only works with multiplies of 16, panics otherwise
fn parallel_dot(x: &[f32], y: &[f32]) -> f32 {
    let res: f32 = x
        .par_chunks(16)
        .map(f32x16::from_slice_unaligned)
        .zip(y.par_chunks(16).map(f32x16::from_slice_unaligned))
        .map(|(a, b)| a * b)
        .sum::<f32x16>()
        .sum();
    res
}

// Fast but ad hoc
fn unrolled_dot_product(x: &[f32], y: &[f32], dim: u64) -> f32 {
    let n = dim as usize;
    let (mut x, mut y) = (&x[..n], &y[..n]);

    let mut sum = 0.0;
    while x.len() >= 16 {
        sum += x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3]
             + x[4] * y[4] + x[5] * y[5] + x[6] * y[6] + x[7] * y[7]
             + x[8] * y[8] + x[9] * y[9] + x[10] * y[10] + x[11] * y[11]
             + x[12] * y[12] + x[13] * y[13] + x[14] * y[14] + x[15] * y[15];
        x = &x[16..];
        y = &y[16..];
    }

    // Take care of any left over elements (if len is not divisible by 8).
    x.iter().zip(y.iter()).fold(sum, |sum, (&ex, &ey)| sum + (ex * ey))
}

// Only works with multiplies of 16
fn efficient_dot(x: &[f32], y: &[f32]) -> f32 {

    let res: f32 = x
        .chunks_exact(16)
        .map(f32x16::from_slice_unaligned)
        .zip(y.chunks_exact(16).map(f32x16::from_slice_unaligned))
        .map(|(a, b)| a * b)
        .sum::<f32x16>()
        .sum();
    res
}

// Add input's values to output
// [1,2,3], [0,1,0] -> [1,2,3], [1,3,3]
fn copy_val(input: &[f32], output: &mut[f32], dim: u64, r: f32) {        
    let n = dim as usize;
    let (mut x, mut y) = (&input[..n], &output[..n]);

    let mut whole_sum_slice = [0.0; DIMENSIONALITY as usize];
    
    while x.len() >= 16 {
        // TODO: multiply all x[i]'s with r
        let sum_slice = &[x[0] * r  + y[0],
                          x[1] * r  + y[1],
                          x[2] * r  + y[2],
                          x[3] * r  + y[3],
                          x[4] * r  + y[4],
                          x[5] * r  + y[5],
                          x[6] * r  + y[6],
                          x[7] * r  + y[7],
                          x[8] * r  + y[8],
                          x[9] * r  + y[9],
                          x[10] * r + y[10],
                          x[11] * r + y[11],
                          x[12] * r + y[12],
                          x[13] * r + y[13],
                          x[14] * r + y[14],
                          x[15] * r + y[15]];

        (&mut whole_sum_slice[..16]).copy_from_slice(sum_slice);
        x = &x[16..];
        y = &y[16..];
    }

    // TODO: take care of the part that's not divisible by 16
    output.copy_from_slice(&whole_sum_slice);
}


fn word2vec(data: Vec<u64>,  data_len: u64, vocab_size: u64, dimensionality: u64) {

    let embedding_len = (vocab_size * dimensionality) as usize;
    let mut word_vectors: Vec<f32> = vec![];
    let mut context_vectors: Vec<f32> = vec![];

    // Populate word vectors with random numbers
    for _ in 0..embedding_len {
        let p1: f32 = rand::random::<f32>() - 0.5;
        let p2: f32 = rand::random::<f32>() - 0.5;

        word_vectors.push(p1);
        context_vectors.push(p2);

    }

    println!("{:?}", word_vectors);

    fn dot_prod(vec1: &[f32], vec2: &[f32], dim: u64) -> f32 {
        //let prod1 = efficient_dot(vec1, vec1);
        let prod2 = unrolled_dot_product(vec1, vec1, dim);
        //let prod3 = parallel_dot(vec1, vec1);
        //println!("Prods: {} {} {}", prod1, prod2, prod3);
        prod2

    }

    let mut word_gradient: Vec<f32> = vec![0.0; embedding_len as usize];
    let mut context_gradient: Vec<f32> = vec![0.0; embedding_len as usize];

    let start = CONTEXT_SIZE;
    let end = data_len -CONTEXT_SIZE;

    let mut rng = thread_rng();

    let mut timestamp = Instant::now();
    for data_pass in 0..PASSES {
        println!("Data pass {}", data_pass);
        for i in start..end {

            if (i + 1) % BATCH_SIZE == 0 {
                println!("Batch {} done! i: {}", i / BATCH_SIZE, i);
                println!("Duration: {} seconds", timestamp.elapsed().as_secs_f32());
                timestamp = Instant::now();
                //println!("{:?}", word_gradient);
                // TODO: priors
                for j in 0..embedding_len {
                    // Add priors to gradient
                    word_gradient[j] = - LAMBDA_0 * word_vectors[j];
                    context_gradient[j] = - LAMBDA_0 * context_gradient[j];

                    // Add to vectors
                    let batch_ratio = (data_len / BATCH_SIZE) as f32;
                    word_vectors[j] += word_gradient[j] * LEARNING_RATE * batch_ratio;
                    context_vectors[j] += context_gradient[j] * LEARNING_RATE * batch_ratio;
                }
                // Flush gradients
                word_gradient = vec![0.0; embedding_len as usize];
                context_gradient = vec![0.0; embedding_len as usize];
                println!("Doned.");
            }
            // Positive samples

            let word_index = data[i as usize];
            
            let word_ix = word_index as usize;
            let dim = dimensionality as usize;
            let word_slice = &word_vectors[word_ix..word_ix+ dim];
            let dot: f32 = (1..CONTEXT_SIZE).map(|separation| {
                let context_ix_after = data[(i + separation) as usize] as usize;
                let context_ix_before = data[(i - separation) as usize] as usize;
                
                let context_slice_after = &context_vectors[context_ix_after..context_ix_after+dim];
                let context_slice_before = &context_vectors[context_ix_before..context_ix_before+dim];

                let dot1 = dot_prod(word_slice, context_slice_after, dimensionality);
                let dot2 = dot_prod(word_slice, context_slice_before, dimensionality);
                dot1 + dot2
            }).sum();
            let sigm = sigmoid(-dot);
            for separation in 1..CONTEXT_SIZE {
                
                let context_ix_after = data[(i + separation) as usize] as usize;
                let context_ix_before = data[(i - separation) as usize] as usize;
                
                let dim = dimensionality as usize;
                let word_ix = word_index as usize;

                let word_slice = &word_vectors[word_ix..word_ix+dim];
                let context_slice_after = &context_vectors[context_ix_after..context_ix_after+dim];
                let context_slice_before = &context_vectors[context_ix_before..context_ix_before+dim];

                let mut word_gradient_slice = &mut word_gradient[word_ix..word_ix+dim];
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
                let mut dot = 0.0;

                for separation in 1..CONTEXT_SIZE {
                    let context_ix_after = data[(ns_i + separation) as usize] as usize;
                    let context_ix_before = data[(ns_i - separation) as usize] as usize;

                    let dim = dimensionality as usize;
                    let context_slice_after = &context_vectors[context_ix_after..context_ix_after+dim];
                    let context_slice_before = &context_vectors[context_ix_before..context_ix_before+dim];

                    dot += dot_prod(word_slice, context_slice_after, dimensionality);
                    dot += dot_prod(word_slice, context_slice_before, dimensionality);
                }
                let sigm = sigmoid(dot);
                for separation in 1..CONTEXT_SIZE {
                    let context_ix_after = data[(ns_i + separation) as usize] as usize;
                    let context_ix_before = data[(ns_i - separation) as usize] as usize;
                    
                    let dim = dimensionality as usize;
                    let word_ix = word_index as usize;

                    let word_slice = &word_vectors[word_ix..word_ix+dim];
                    let context_slice_after = &context_vectors[context_ix_after..context_ix_after+dim];
                    let context_slice_before = &context_vectors[context_ix_before..context_ix_before+dim];

                    let mut word_gradient_slice = &mut word_gradient[word_ix..word_ix+dim];
                    copy_val(&context_slice_after, &mut word_gradient_slice, dimensionality, -sigm);
                    copy_val(&context_slice_before, &mut word_gradient_slice, dimensionality, -sigm);

                    let mut context_gradient_slice_after = &mut context_gradient[context_ix_after..context_ix_after+dim];
                    copy_val(&word_slice, &mut context_gradient_slice_after, dimensionality, -sigm);
                    let mut context_gradient_slice_before = &mut context_gradient[context_ix_before..context_ix_before+dim];

                    copy_val(&word_slice, &mut context_gradient_slice_before, dimensionality, -sigm);

                }
            }
        }
    }

    println!("{:?}", &word_vectors[..100]);
    write_embeddings(word_vectors, context_vectors);

}


fn main() {
    let args: Vec<String> = env::args().collect();
    let dataset_name: String = args[1].to_string();
    //generate_data(dataset_name);

    println!("Generate random data...");
    let mut rng = thread_rng();
    let data_len = 1000000 as u64;
    let vocab_size = 3700;
    let data: Vec<u64> = (0..data_len).map(|i| {
        if i % 100000 == 0 {
            println!("i: {}", i);
        }
        rng.gen_range(0, vocab_size)
    }).collect();

    println!("Done.");

    word2vec(data, data_len, vocab_size, DIMENSIONALITY);

    println!("Traineds.");
}
