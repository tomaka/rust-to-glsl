#![feature(phase)]

#[phase(plugin)]
extern crate rust_to_glsl;

fn main() {
    let glsl = to_glsl!(
        const x: uint = 5;

        static t: f32 = 1;

        fn hello() {
            min(5, 3)
        }
    );

    println!("{}", glsl);
}
