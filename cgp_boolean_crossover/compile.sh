#!/bin/bash

cargo build --features "tournament standard" --release --target-dir tournament_vanilla
cargo build --features "tournament ereorder" --release --target-dir tournament_ereorder
cargo build --features "mulambda_crossover standard" --release --target-dir mulambda_vanilla
cargo build --features "mulambda_crossover ereorder" --release --target-dir mulambda_ereorder
cargo build --features "mulambda standard" --release --target-dir standard
cargo build --features "mulambda ereorder" --release --target-dir standard_ereorder
