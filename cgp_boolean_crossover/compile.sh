#!/bin/bash

cargo build --features "tournament vanilla" --release --target-dir tournament_vanilla
cargo build --features "tournament ereorder" --release --target-dir tournament_ereorder
cargo build --features "mulambda_crossover vanilla" --release --target-dir mulambda_vanilla
cargo build --features "mulambda_crossover ereorder" --release --target-dir mulambda_ereorder
cargo build --features "mulambda vanilla" --release --target-dir standard
cargo build --features "mulambda ereorder" --release --target-dir standard_ereorder