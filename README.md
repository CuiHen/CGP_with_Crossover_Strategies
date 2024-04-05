# CGP With Crossover
Code and Benchmarks for Positional Bias Does Not Influence Cartesian Genetic Programming with Crossover

# Rust
The code is written in Rust only.  
For installation, see: https://github.com/rust-lang/rust/blob/master/README.md

# Building
You have to build everything yourself. You will need a working `Rust` and `Cargo` setup. [Rustup](https://rustup.rs/) is the simplest way to set this up on either Windows, Mac or Linux.

Once the prerequisites have been installed, compilation on your native platform is as simple as running the following in a terminal:

```
cargo build --release --features "FEATURE1 FEATURE2"
```
`FEATURE1` is the selection method:
- tournament: a standard tournament selection with elitists
- mulambda_crossover: a standard (mu + lambda)-ES; mainly used in combination with a crossover operator
- mulambda: defines the (1+4)-ES

`FEATURE2` is the CGP version:
- standard: the baseline CGP implementation without extensions
- ereorder: CGP with the E-Reorder extension (see: DOI: 10.5220/0012174100003595)


# Usage
Run the build executable on your machine via:
```
./target/release/cgp
```
or 
```
./target/release/cgp.exe
```

Outputs will be placed into a folder called
`Experiments_Output`

You can configure the run via following command line arguments:
- `run-id`
  - The ID of the run
  - Only important for saving results
  - default: 0
- `dataset`
  - which dataset to use. For Boolean:
    - 0: Parity
    - 1: Encode
    - 2: Decode
    - 3: Multiply
  - for symbolic regression:
    - 0: nguyen_7
    - 1: koza_3
    - 2: pagie_1
    - 3: keijzer_6
- `nbr-nodes`
  - the number of computational nodes for CGP
- `cgp-type`
  - the CGP type used
    - 0: "Standard"
    - 1: "Reorder_Equidistant"
  - Only important for saving results
- `crossover-type`
  - 0: 1-point crossover
  - 1: multi-n crossover
  - 2: uniform crossover
  - 3: no crossover
- `multi-point-n`
  - only used if `crossover-type` == 1
  - number of `n` for `multi-n crossover`
  - i.e.: `multi-point-n` == 2, then: 2-point crossover
  - i.e.: `multi-point-n` == 3, then: 3-point crossover  
- `crossover-rate`
- `tournament-size`
- `elitism-number`
  - number of elitists that will be included into the population 
- `population-size`
- `runner-type`
  - Only important for saving results.
  - 0: (1+4)-ES
  - 1: (mu + lambda)-ES
  - 2: tournament selection

## Important Note
If a (1+4)-ES is used, the arguments:
  - crossover-type
  - multi-point-n
  - crossover-rate
  - tournament-size
  - elitism-number
  - population-size
will be ignored and does not have to be filled out.

If a (mu+lambda)-ES is used, the arguments:
  - tournament-size
will be ignored.
Furthermore:
  - mu is equivalent to `elitism-number`
  - lambda is equivalent to `population-size`

