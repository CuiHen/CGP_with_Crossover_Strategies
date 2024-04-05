use float_eq::float_eq;
use std::io::Write;
use cgp_regression_crossover::global_params::CgpParameters;
use cgp_regression_crossover::datasets::*;
use clap::Parser;
use std::fs;
use std::fs::File;
use std::path::Path;
use std::process::exit;
use rayon::prelude::*;


#[cfg(feature = "mulambda")]
use cgp_regression_crossover::utils::runner::Runner;
#[cfg(feature = "tournament")]
use cgp_regression_crossover::utils::runner_multiple_parents_with_elitist_tournament::Runner;
#[cfg(feature = "mulambda_crossover")]
use cgp_regression_crossover::utils::runner_multiple_parents_with_elitist_mulambda::Runner;

#[derive(Parser, Clone)]
#[clap(author, version, about, name = "testname")]
struct Args {
    #[arg(long, default_value_t = 0)]
    run_id: usize,

    #[arg(long, default_value_t = 0)]
    dataset: usize,

    #[arg(long, default_value_t = 450)]
    nbr_nodes: usize,

    //         0 => format!("Baseline_standard"),
    //         1 => format!("ereorder"),
    #[arg(long, default_value_t = 1)]
    cgp_type: usize,

    // 0: point crossover
    // 1: multi-n crossover
    // 2: uniform crossover
    // 3: no crossover
    #[arg(long, default_value_t = 2)]
    crossover_type: usize,

    #[arg(long, default_value_t = 0.9)]
    crossover_rate: f32,

    #[arg(long, default_value_t = 4)]
    tournament_size: usize,

    #[arg(long, default_value_t = 10)]
    elitism_number: usize,

    #[arg(long, default_value_t = 20)]
    population_size: usize,

    // for n-point crossover
    #[arg(long, default_value_t = 3)]
    multi_point_n: usize,

    #[arg(long, default_value_t = 1)]
    runner_type: usize,
}


fn main() {
    // ################################################################################
    // ############################ Arguments #########################################
    // ################################################################################
    let args = Args::parse();

    let ((data, label),
        (eval_data, eval_label)) = match args.dataset {
        0 => (keijzer::get_dataset(), keijzer::get_eval_dataset()),
        1 => (koza_3::get_dataset(), koza_3::get_eval_dataset()),
        2 => (nguyen_7::get_dataset(), nguyen_7::get_eval_dataset()),
        3 => (pagie_1::get_dataset(), pagie_1::get_eval_dataset()),
        _ => panic!("Wrong dataset"),
    };

    let mut params = CgpParameters::default();

    let nbr_inputs = data[0].len();
    let nbr_outputs = 1;

    params.nbr_inputs = nbr_inputs;
    params.nbr_outputs = nbr_outputs;
    params.nbr_computational_nodes = args.nbr_nodes;
    params.crossover_type = args.crossover_type;
    params.crossover_rate = args.crossover_rate;
    params.tournament_size = args.tournament_size;
    params.elitism_number = args.elitism_number;
    params.multi_point_n = args.multi_point_n;
    params.population_size = args.population_size;
    params.cgp_type = args.cgp_type;

    // ################################################################################
    // ############################ Logger ############################################
    // ################################################################################

    let runner_type = match args.runner_type {
        0 => format!("one_plus_four"),
        1 => format!("mu_{}_lambda_{}", args.elitism_number, args.population_size),
        2 => format!("tournament_pop_size_{}_t_size_{}_elitism_{}", args.population_size, args.tournament_size, args.elitism_number),
        _ => panic!("wrong runner type"),
    };

    let dataset_string = match args.dataset {
        0 => "keijzer_6",
        1 => "koza_3",
        2 => "nguyen_z",
        3 => "pagie_1",
        _ => panic!("Wrong dataset"),
    };

    let cgp_type_string = match args.cgp_type {
        0 => format!("Baseline_standard"),
        1 => format!("ereorder"),
        _ => panic!("Wrong type"),
    };

    let crossover_type = match args.crossover_type {
        0 => format!("point_crossover"),
        1 => format!("multi_{}_crossover", args.multi_point_n),
        2 => format!("uniform_crossover"),
        3 => format!("no_crossover"),
        _ => panic!("Wrong type"),
    };

    let save_path = Path::new("")
        .join("Experiments_Output_regression")
        .join(runner_type)
        .join(cgp_type_string)
        .join(crossover_type)
        .join(dataset_string)
        .join(format!("number_nodes_{}_{}", args.nbr_nodes, "single"));

    fs::create_dir_all(save_path.clone()).unwrap();

    // ################################################################################
    // ############################ Training ##########################################
    // ################################################################################
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();


    let save_file_iteration = format!("run_{}_iteration.txt", args.run_id);
    let mut output_file = File::create(save_path.join(save_file_iteration))
        .expect("cannot create file");

    let mut runner = Runner::new(params.clone(),
                                 data,
                                 label,
                                 eval_data,
                                 eval_label);
    let mut runtime: usize = 0;
    let mut fitness_eval = f32::MAX;
    let mut fitness_train = f32::MAX;

    while runtime < 500_000 {
        writeln!(output_file, "Iteration: {runtime}, Fitness: {:?}", runner.get_best_fitness()).expect("write not okay??");
        // if runtime % params.eval_after_iterations == 0 {
        //     writeln!(lock, "Iteration: {runtime}, Fitness: {:?}", runner.get_best_fitness()).expect("write not okay??");
        // }

        runner.learn_step(runtime);  // lern step

        runtime += 1;

        if float_eq!(runner.get_best_fitness(), 0., abs <= 0.01) {  // for single parent
            break;
        }
    }
    fitness_eval = runner.get_test_fitness();
    fitness_train = runner.get_best_fitness();

    // ################################################################################
    // ############################ Saving to text ####################################
    // ################################################################################
    println!("{runtime}");

    writeln!(output_file, "End at iteration: {}", runtime).expect("cannot write");
    writeln!(output_file, "Fitness Eval: {}", fitness_eval).expect("cannot write");
    writeln!(output_file, "Fitness Train: {}", fitness_train).expect("cannot write");

    let save_file_active_node = format!("run_{}_active_node.txt", args.run_id);
    let mut output = File::create(save_path.join(save_file_active_node))
        .expect("cannot create file");

    let mut parent = runner.get_parent();
    parent.get_active_nodes_id();

    write!(output, "{:?}", parent.active_nodes.unwrap()).expect("cannot write");
}
