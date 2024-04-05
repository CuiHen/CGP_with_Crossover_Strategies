use std::cmp::min;
use rand;
use rand::Rng;
use rand::distributions::{Distribution, Uniform};
use rand::prelude::IteratorRandom;

#[cfg(feature = "tournament")]
use crate::utils::runner_multiple_parents_with_elitist_tournament::Runner;
#[cfg(feature = "mulambda_crossover")]
use crate::utils::runner_multiple_parents_with_elitist_mulambda::Runner;

#[cfg(feature = "standard")]
use crate::standard_cgp::chromosome::Chromosome;
#[cfg(feature = "ereorder")]
use crate::reorder::chromosome_reorder_equidistant::Chromosome;


pub fn single_point_crossover(runner: &mut Runner,
                              new_population: &mut Vec<Chromosome>,
                              child1_id: usize,
                              child2_id: usize,
                              parent1_id: usize,
                              parent2_id: usize) {
    // Generate range between computational nodes
    let crossover_point = runner.rng.gen_range(runner.params.nbr_inputs..runner.params.nbr_inputs + runner.params.nbr_computational_nodes);

    let mut cross_chromo_1: Chromosome = runner.population[parent1_id].clone();
    let mut cross_chromo_2: Chromosome = runner.population[parent2_id].clone();

    cross_chromo_1
        .nodes_grid[..crossover_point]
        .swap_with_slice(&mut cross_chromo_2.nodes_grid[..crossover_point]);

    cross_chromo_1.get_active_nodes_id();
    cross_chromo_2.get_active_nodes_id();

    new_population[child1_id] = cross_chromo_1;
    new_population[child2_id] = cross_chromo_2;
}


pub fn multi_point_crossover(runner: &mut Runner,
                             new_population: &mut Vec<Chromosome>,
                             child1_id: usize,
                             child2_id: usize,
                             parent1_id: usize,
                             parent2_id: usize) {
    // Generate range between computational nodes
    let mut rng = rand::thread_rng();

    let mut cross_chromo_1: Chromosome = runner.population[parent1_id].clone();
    let mut cross_chromo_2: Chromosome = runner.population[parent2_id].clone();

    let crossover_points: Vec<usize> = (runner.params.nbr_inputs..runner.params.nbr_inputs + runner.params.nbr_computational_nodes)
        .choose_multiple(&mut rng, runner.params.multi_point_n);

    for point in crossover_points {
        cross_chromo_1.nodes_grid[point..].swap_with_slice(&mut cross_chromo_2.nodes_grid[point..]);
    }

    cross_chromo_1.get_active_nodes_id();
    cross_chromo_2.get_active_nodes_id();

    new_population[child1_id] = cross_chromo_1;
    new_population[child2_id] = cross_chromo_2;
}


pub fn uniform_crossover(runner: &mut Runner,
                         new_population: &mut Vec<Chromosome>,
                         child1_id: usize,
                         child2_id: usize,
                         parent1_id: usize,
                         parent2_id: usize) {
    let mut between = Uniform::from(0..=1);
    let mut rng = rand::thread_rng();

    let mut cross_chromo_1: Chromosome = runner.population[parent1_id].clone();
    let mut cross_chromo_2: Chromosome = runner.population[parent2_id].clone();

    for node_id in runner.params.nbr_inputs..runner.params.nbr_inputs + runner.params.nbr_computational_nodes {
        let cross = between.sample(&mut rng);

        if cross == 0 {
            std::mem::swap(&mut cross_chromo_1.nodes_grid[node_id], &mut cross_chromo_2.nodes_grid[node_id])
        }
    }

    cross_chromo_1.get_active_nodes_id();
    cross_chromo_2.get_active_nodes_id();

    new_population[child1_id] = cross_chromo_1;
    new_population[child2_id] = cross_chromo_2;
}

pub fn no_crossover(runner: &mut Runner,
                         new_population: &mut Vec<Chromosome>,
                         child1_id: usize,
                         child2_id: usize,
                         parent1_id: usize,
                         parent2_id: usize) {

    new_population[child1_id] = runner.population[parent1_id].clone();
    new_population[child2_id] = runner.population[parent2_id].clone();

}
// pub fn subgraph_crossover(runner: &mut Runner,
//                           new_population: &mut Vec<Chromosome>,
//                           child1_id: usize,
//                           child2_id: usize,
//                           parent1_id: usize,
//                           parent2_id: usize) {
//
//     // im Paper: behind cp == alles von cp bis output nodse
//     // in front of cp == input nodes bis cp
//
//     // ## Preliminary
//     // get number of active ndoes
//     let active_nodes_p1 = runner.population[parent1_id].active_nodes.clone().unwrap().clone();
//     let active_nodes_p2 = runner.population[parent2_id].active_nodes.clone().unwrap();
//     let mut in_out_nodes: Vec<usize> = (0..runner.params.nbr_inputs).collect();
//     let output_nodes: Vec<usize> = (runner.params.nbr_inputs + runner.params.nbr_computational_nodes
//         ..
//         runner.params.nbr_inputs + runner.params.nbr_computational_nodes + runner.params.nbr_outputs)
//         .collect();
//
//     in_out_nodes.extend(output_nodes);
//     // remove input and output nodes
//     let active_comp_nodes_p1 = vect_difference(&active_nodes_p1, &in_out_nodes);
//     let active_comp_nodes_p2 = vect_difference(&active_nodes_p2, &in_out_nodes);
//
//     if (active_comp_nodes_p1.len() < 2) | (active_comp_nodes_p2.len() < 2) {
//         new_population[child1_id] = runner.population[parent1_id].clone();
//         new_population[child2_id] = runner.population[parent2_id].clone();
//         return;
//     }
//
//     // let len_active_node_p1 = active_nodes_p1.len();
//     // let len_active_node_p2 = active_nodes_p2.len();
//
//     for child_id in [child1_id, child2_id] {
//         //     ## Step 0: Preliminary
//         // define crossover points
//         // let cp1 = runner.rng.gen_range(active_nodes_p1.first()..active_nodes_p1.last());
//         let cp1 = runner.rng.gen_range(*active_comp_nodes_p1.iter().min().unwrap()..*active_comp_nodes_p1.iter().max().unwrap());
//         let cp2 = runner.rng.gen_range(*active_comp_nodes_p2.iter().min().unwrap()..*active_comp_nodes_p2.iter().max().unwrap());
//
//         //     ## Step 1: Define a general crossover point
//         let cp = min(cp1, cp2);
//
//         //     ## Step 2: Copy genetic material in front of the crossover point
//         // case cp1 is min: beginning of parent1 stays the same
//         let mut cross_chromo: Chromosome;
//         if cp1 < cp2 {
//             cross_chromo = runner.population[parent1_id].clone();
//             cross_chromo.nodes_grid[cp..]
//                 .clone_from_slice(&runner.population[parent2_id].nodes_grid[cp..]);
//         } else {
//             // else: beginning of parent2 stays the same
//             cross_chromo = runner.population[parent2_id].clone();
//             cross_chromo.nodes_grid[cp..]
//                 .clone_from_slice(&runner.population[parent1_id].nodes_grid[cp..]);
//         }
//         //  ## Step 3: connect both sections
//
//
//         //     ## Done:
//         cross_chromo.get_active_nodes_id();
//         new_population[child_id] = cross_chromo;
//     }
// }






















