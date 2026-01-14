#!/usr/bin/env rust-script
//! Generate Swift and Kotlin bindings for AgentVec mobile
//!
//! Usage:
//!   cargo run --manifest-path generate-bindings.rs

use std::fs;
use std::path::Path;

fn main() {
    let udl_path = "src/agentvec.udl";
    let swift_out = "bindings/swift";
    let kotlin_out = "bindings/kotlin";

    // Create output directories
    fs::create_dir_all(swift_out).expect("Failed to create Swift output directory");
    fs::create_dir_all(kotlin_out).expect("Failed to create Kotlin output directory");

    println!("To generate bindings manually:");
    println!("\n1. Install uniffi-bindgen:");
    println!("   cargo install uniffi-bindgen");
    println!("\n2. Generate Swift bindings:");
    println!("   uniffi-bindgen generate {} --language swift --out-dir {}", udl_path, swift_out);
    println!("\n3. Generate Kotlin bindings:");
    println!("   uniffi-bindgen generate {} --language kotlin --out-dir {}", udl_path, kotlin_out);
    println!("\nOr use the uniffi CLI if available in your version:");
    println!("   uniffi-bindgen generate src/agentvec.udl --language swift");
    println!("   uniffi-bindgen generate src/agentvec.udl --language kotlin");
}
