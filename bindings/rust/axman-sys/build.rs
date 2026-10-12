use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // axman root is 3 levels up: axman-sys -> rust -> bindings -> axman
    let axman_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .expect("Failed to locate axman root");

    // Link against libaxman in axman root or build folder
    let build_dir = axman_root.join("build");
    
    println!("cargo:rustc-link-search=native={}", axman_root.display());
    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=dylib=axman");

    // On Windows, copy libaxman.dll into target directory so tests can find it
    if cfg!(windows) {
        let dll_src = if axman_root.join("libaxman.dll").exists() {
            axman_root.join("libaxman.dll")
        } else {
            build_dir.join("libaxman.dll")
        };
        if dll_src.exists() {
            let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
            // Target dir is out_dir/../../../
            if let Some(target_dir) = out_dir.ancestors().nth(3) {
                let deps_dir = target_dir.join("deps");
                let _ = std::fs::copy(&dll_src, target_dir.join("libaxman.dll"));
                let _ = std::fs::copy(&dll_src, deps_dir.join("libaxman.dll"));
            }
        }
    }

    println!("cargo:rerun-if-changed={}", axman_root.join("include").join("axman_c_api.h").display());
}
