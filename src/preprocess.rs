use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::{self, BufRead};
use std::error::Error;
use std::io::prelude::*;
use std::path::Path;
use ndarray::{Array, Array1};
use ndarray_npy::{WriteNpyError, WriteNpyExt};

// PREPROCESSING CONSTS
const SKIP_LIMIT: u64 = 5;
const THRESHOLD: f64 = 0.0001;


fn write_vector(vector: Vec<u64>) -> Result<(), WriteNpyError> {
    let arr: Array1<u64> = Array::from(vector);
    let writer = File::create("data.npy")?;
    arr.write_npy(writer)?;
    Ok(())
}

fn write_unigram(inverse_vocabulary: HashMap<u64, String>, vocab_size: u64) {
    let path = Path::new("unigram.txt");
    let display = path.display();

    let mut s = "".to_string();
    for ix in 0..vocab_size {
        //println!("Moi {} {}", ix, inverse_vocabulary[&ix]);
        s += &format!("{} {}\n", ix, inverse_vocabulary[&ix]);
    }
    // Open a file in write-only mode, returns `io::Result<File>`
    let mut file = match File::create(&path) {
        Err(why) => panic!("couldn't create {}: {}", display, why.description()),
        Ok(file) => file,
    };

    match file.write_all(s.as_bytes()) {
        Err(why) => panic!("couldn't write to {}: {}", display, why.description()),
        Ok(_) => println!("successfully wrote to {}", display),
    }

}
pub fn write_embeddings(word: Vec<f32>, context: Vec<f32>) -> Result<(), WriteNpyError> {
    let word_arr: Array1<f32> = Array::from(word);
    let context_arr: Array1<f32> = Array::from(context);
    let word_writer = File::create("word.npy")?;
    word_arr.write_npy(word_writer)?;
    let context_writer = File::create("context.npy")?;
    context_arr.write_npy(context_writer)?;
    Ok(())
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

fn discard(current: u64, total: u64) -> bool {
	let ratio = THRESHOLD / ( (current as f64) / (total as f64) );
    let p_0 = 1.0 - ratio.sqrt();
    let p: f64 = rand::random::<f64>();
    p < p_0
}

fn clean_string(input_string: String) -> String {
    input_string
        .replace(|c: char| !c.is_alphanumeric(), "")
        .to_lowercase()
}

pub fn generate_data(dataset_name: String) -> Vec<u64> {
    println!("Fetch data from {}*", dataset_name);
	// Fetch dataset name from arguments

    let mut vocabulary: HashMap<String, u64> = HashMap::new();
    let mut inverse_vocabulary: HashMap<u64, String> = HashMap::new();
    let mut vocab_size = 0;

    let mut word_count: HashMap<u64, u64> = HashMap::new();

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

    					let vocab_index = vocabulary.entry(token.to_string()).or_insert(vocab_size);
    					if *vocab_index == vocab_size {
                            inverse_vocabulary.insert(vocab_size,token);
    						vocab_size += 1;
    					}
                        
                        data.push(*vocab_index);

                        let count = word_count.entry(vocab_index.clone()).or_insert(0);
                        *count += 1;
                    }
                }
            }
        }
    }
    
    let mut i = 0;
    for (key, value) in vocabulary.clone() {
        if i > 25 {
            break;
        } else {
            i += 1;
        }
        println!("{} -> {}", key, value);
    }

    println!();
    
    i = 0;
    for (key, value) in word_count.clone() {
        
        let word = inverse_vocabulary.clone()[&key].clone();
        
        if i < 25 {
            println!("{} ({}) -> {}", word, key, value);
            i += 1;
        } else {
            break;
        }
    }

    let data_len = data.len() as u64;
    let mut filtered_vocab_size = 0;
    let mut filtered_data: Vec<u64> = vec![];
    let mut filtered_vocabulary: HashMap<String, u64> = HashMap::new();
    let mut inverse_filtered_vocabulary: HashMap<u64, String> = HashMap::new();

    for (ix, word_type_index) in data.iter().enumerate() {
        let word_occurences = word_count[&word_type_index];
        let word_str = inverse_vocabulary[&word_type_index].clone();

        if ix % 10000 == 0 {
            println!("ix: {} ({})", ix, word_str);
        }
        if word_occurences >= SKIP_LIMIT {
            let vocab_index = filtered_vocabulary.entry(word_str.to_string()).or_insert(filtered_vocab_size);
            if *vocab_index == filtered_vocab_size {
                inverse_filtered_vocabulary.insert(filtered_vocab_size, word_str.clone());
                filtered_vocab_size += 1;
            }
            if !discard(word_occurences, data_len) {
                filtered_data.push(vocab_index.clone());
            } else {
                //println!("Skip: {}", word_str);
            }
        }

    }

    let filtered_data_len = filtered_data.len();
    for n in 0..filtered_vocab_size {
        println!("Index: {}, word {}", n, inverse_filtered_vocabulary[&n]);

    }
    println!("Vocabulary len: {}", vocabulary.len());
    println!("Filtered vocabulary len: {}", filtered_vocabulary.len());
    println!("Inverse vocabulary: {:?}", inverse_vocabulary.len());
    println!("Word count: {:?}", word_count.len());

    println!("Filtered data len: {:?}", &filtered_data[..25]);
    println!("Filtered data len: {}", filtered_data_len);
    let _ = write_vector(filtered_data.clone());
    write_unigram(inverse_filtered_vocabulary, filtered_vocab_size);
    filtered_data
}

