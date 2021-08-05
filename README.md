# Crate `lstsq` for the [Rust language](https://www.rust-lang.org/)

[![Crates.io](https://img.shields.io/crates/v/lstsq.svg)](https://crates.io/crates/lstsq)
[![Documentation](https://docs.rs/lstsq/badge.svg)](https://docs.rs/lstsq/)
[![Crate License](https://img.shields.io/crates/l/lstsq.svg)](https://crates.io/crates/lstsq)
[![Dependency status](https://deps.rs/repo/github/strawlab/lstsq/status.svg)](https://deps.rs/repo/github/strawlab/lstsq)
[![build](https://github.com/strawlab/lstsq/workflows/build/badge.svg?branch=main)](https://github.com/strawlab/lstsq/actions?query=branch%3Amain)

Return the least-squares solution to a linear matrix equation

## About

The crate implements the linear least squares solution to a linear matrix
equation.

Characteristics:

* Linear algebra and types from the [`nalgebra`](https://docs.rs/nalgebra)
  crate.
* Maximum compatibility with the
  [`numpy.linalg.lstsq`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html)
  Python library function.
* No standard library is required (disable the default features to disable
  use of `std`) and no heap allocations. In other words, this can run on a
  bare-metal microcontroller with no OS.

## Testing

### Unit tests

To run the unit tests:

```
cargo test
```

### Test for `no_std`

Since the `thumbv7em-none-eabihf` target does not have `std` available, we
can build for it to check that our crate does not inadvertently pull in
std. The unit tests require std, so cannot be run on a `no_std` platform.
The following will fail if a std dependency is present:

```
# install target with: "rustup target add thumbv7em-none-eabihf"
cargo build --no-default-features --target thumbv7em-none-eabihf
```
