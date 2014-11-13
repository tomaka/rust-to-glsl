#![feature(phase)]

#[phase(plugin)]
extern crate rust_to_glsl;

fn main() {
    let glsl = to_glsl!(
        const x: uint = 5;

        static texture: &Texture2d = 1;

        fn hello() {
            min(5, 3 * 1 + 5)
        }
    );

    println!("{}", glsl);
}
