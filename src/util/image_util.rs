use std::fs;

pub fn list_files(floder: &str) {
    for file in fs::read_dir(floder).unwrap() {
        match file {
            Ok(f) => {
                println!("image name is {}", f.path().display())
            },
            Err(e) => {}
        }
    }
} 