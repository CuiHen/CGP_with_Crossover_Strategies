// use std::collections::HashSet;
use std::fmt::{Display, Formatter};
use rand;
use ndarray::Array2;
use rand::prelude::IteratorRandom;
use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use crate::global_params::CgpParameters as g_params;
use crate::utils::utility_funcs;
use crate::utils::utility_funcs::{get_argmin, get_argmins_of_value, vect_difference};
use crate::utils::crossover::crossover_algos;

#[cfg(feature = "standard")]
use crate::standard_cgp::chromosome::Chromosome;
#[cfg(feature = "ereorder")]
use crate::reorder::chromosome_reorder_equidistant::Chromosome;


pub struct Runner {
    pub params: g_params,
    pub data: Array2<bool>,
    pub label: Array2<bool>,
    pub population: Vec<Chromosome>,
    pub fitness_vals_sorted: Vec<f32>,
    // check for correctness, must include elitists too
    pub fitness_vals: Vec<f32>,
    // check for correctness, must include elitists too
    pub tournament_selected: Vec<usize>,
    pub rng: ThreadRng,

    pub elitist_ids: Vec<usize>,
    pub child_ids: Vec<usize>,
}

impl Display for Runner {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Fitnesses: {:?}", self.fitness_vals)
    }
}

impl Runner {
    pub fn new(params: g_params, data: Array2<bool>, label: Array2<bool>) -> Self {
        // let mut rng = ChaCha8Rng::seed_from_u64(1050);
        let mut rng = rand::thread_rng();


        let mut population: Vec<Chromosome> = Vec::with_capacity(params.population_size + params.elitism_number);
        let mut fitness_vals: Vec<f32> = Vec::with_capacity(params.population_size + params.elitism_number);

        for _ in 0..(params.population_size + params.elitism_number) {
            let mut chromosome = Chromosome::new(params.clone());
            let fitness = chromosome.evaluate(&data, &label);

            fitness_vals.push(fitness);
            population.push(chromosome);
        }

        // Get sorted fitness vals
        let mut fitness_vals_sorted = fitness_vals.clone();
        fitness_vals_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Reverse fitness_vals_sorted to pop the best fitness first
        let mut temp_fitness_vals_sorted = fitness_vals_sorted.clone();
        temp_fitness_vals_sorted.reverse();
        temp_fitness_vals_sorted.dedup();

        let mut elitist_ids: Vec<usize> = vec![];

        while elitist_ids.len() < params.elitism_number {
            let current_best_fitness_val = temp_fitness_vals_sorted.pop().unwrap();

            get_argmins_of_value(&fitness_vals,
                                 &mut elitist_ids,
                                 current_best_fitness_val);
        }

        elitist_ids.truncate(params.elitism_number);


        let child_ids: Vec<usize> = (0..(params.population_size + params.elitism_number)).collect();
        let child_ids = vect_difference(&child_ids, &elitist_ids);


        Self {
            params,
            data,
            label,
            population,
            fitness_vals,
            fitness_vals_sorted,
            rng,
            elitist_ids,
            tournament_selected: vec![],
            child_ids
        }
    }

    pub fn learn_step(&mut self, i: usize) {
        self.get_child_ids();

        self.tournament_selection();

        self.reorder();

        self.crossover();

        self.mutate_chromosomes();

        self.eval_chromosomes();

        self.get_elitists();
    }

    fn get_child_ids(&mut self) {
        // elitists should not be reordered as they did not change
        let child_ids: Vec<usize> = (0..(self.params.population_size + self.params.elitism_number)).collect();
        let child_ids = vect_difference(&child_ids, &self.elitist_ids);

        self.child_ids = child_ids;
    }

    fn reorder(&mut self) {
        for id in 0..(self.params.population_size + self.params.elitism_number) {
            self.population[id].reorder();
        }
    }


    fn tournament_selection(&mut self) {
        let mut selection = vec![];

        // take pop-size many, as pop-size many will be replaced
        for _ in 0..self.params.population_size {
            let winner_id = self.fitness_vals
                .clone()
                .into_iter()
                .enumerate() // get tuples: (i, fitness_val) with i := chromosome id
                .choose_multiple(&mut self.rng, self.params.tournament_size)
                .into_iter()
                .min_by(|i, j| i.1.partial_cmp(&j.1).unwrap())  // Sort by fitness val
                .map(|(i, j)| i)// get id of chromosome
                .unwrap();

            selection.push(winner_id)
        }

        self.tournament_selected = selection;
    }

    fn mutate_chromosomes(&mut self) {
        // get intersection of population ids and elitists
        // let mutation_set: Vec<usize> = (0..(self.params.population_size + self.params.elitism_number)).collect();
        // let mutation_set = vect_difference(&mutation_set, &self.elitist_ids);

        // mutate new chromosomes; do not mutate elitists
        // for id in mutation_set {
        for id in &self.child_ids {
            self.population[*id].mutate_single();
        }
    }

    fn eval_chromosomes(&mut self) {
        // get intersection of population ids and elitists
        // let eval_set: Vec<usize> = (0..(self.params.population_size + self.params.elitism_number)).collect();
        // let eval_set: Vec<usize> = vect_difference(&eval_set, &self.elitist_ids);

        // for id in eval_set {
        for id in &self.child_ids {
            let fitness = self.population[*id].evaluate(&self.data, &self.label);
            self.fitness_vals[*id] = fitness;
        }

        let mut best_fitnesses_sorted = self.fitness_vals.clone();
        best_fitnesses_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        self.fitness_vals_sorted = best_fitnesses_sorted;
    }

    fn get_elitists(&mut self) {
        let mut temp_fitness_vals_sorted = self.fitness_vals_sorted.clone();
        // reverse to pop the last element - the best one
        temp_fitness_vals_sorted.reverse();
        temp_fitness_vals_sorted.dedup();

        let mut elitist_ids: Vec<usize> = vec![];

        while elitist_ids.len() < self.params.elitism_number {
            let current_best_fitness_val = temp_fitness_vals_sorted.pop().unwrap();

            get_argmins_of_value(&self.fitness_vals,
                                 &mut elitist_ids,
                                 current_best_fitness_val);
        }

        elitist_ids.truncate(self.params.elitism_number);
        self.elitist_ids = elitist_ids;
    }

    pub fn get_best_fitness(&self) -> f32 {
        return self.fitness_vals_sorted[0];
    }

    pub fn get_elitism_fitness(&self) -> Vec<f32> {
        let mut results: Vec<f32> = Vec::with_capacity(self.params.elitism_number);
        for id in &self.elitist_ids {
            results.push(self.fitness_vals[*id]);
        }
        return results;
    }

    pub fn get_best_solution(&self) -> Chromosome {
        let idx = get_argmin(&self.fitness_vals);
        return self.population[idx].clone();
    }

    fn crossover(&mut self) {
        // get all new children ids; i.e. the ID's of chromosomes in the population that
        // can be replaced.
        // It must exclude the elitists, otherwise they may be replaced too
        let children_set: Vec<usize> = (0..(self.params.population_size + self.params.elitism_number)).collect();
        let children_set: Vec<usize> = vect_difference(&children_set, &self.elitist_ids);

        // create new population
        let mut new_population: Vec<Chromosome> = self.population.clone();

        for (i, child_ids) in children_set.chunks(2).enumerate() {
            let crossover_prob = rand::random::<f32>();
            if crossover_prob <= self.params.crossover_rate {
                match self.params.crossover_type {
                    0 => crossover_algos::single_point_crossover(self,
                                                                 &mut new_population,
                                                                 child_ids[0],
                                                                 child_ids[1],
                                                                 self.tournament_selected[2 * i],
                                                                 self.tournament_selected[2 * i + 1]),
                    1 => crossover_algos::multi_point_crossover(self,
                                                                &mut new_population,
                                                                child_ids[0],
                                                                child_ids[1],
                                                                self.tournament_selected[2 * i],
                                                                self.tournament_selected[2 * i + 1]),
                    2 => crossover_algos::uniform_crossover(self,
                                                            &mut new_population,
                                                            child_ids[0],
                                                            child_ids[1],
                                                            self.tournament_selected[2 * i],
                                                            self.tournament_selected[2 * i + 1]),
                    3 => crossover_algos::no_crossover(self,
                                                       &mut new_population,
                                                       child_ids[0],
                                                       child_ids[1],
                                                       self.tournament_selected[2 * i],
                                                       self.tournament_selected[2 * i + 1]),
                    _ => panic!("not implemented crossover tpye")
                }
            } else {
                //     no crossover, just copy parents
                new_population[child_ids[0]] = self.population[self.tournament_selected[2 * i]].clone();
                new_population[child_ids[1]] = self.population[self.tournament_selected[2 * i + 1]].clone();
            }
        }
        self.population = new_population;
    }


    fn _deprecated_and_buggy_crossover(&mut self) {
        // get all new children ids; i.e. the ID's of chromosomes in the population that
        // can be replaced.
        // It must exclude the elitists, otherwise they may be replaced too
        let children_set: Vec<usize> = (0..(self.params.population_size + self.params.elitism_number)).collect();
        let children_set: Vec<usize> = vect_difference(&children_set, &self.elitist_ids);

        // create new population
        let mut new_population: Vec<Chromosome> = self.population.clone();
        // let mut i:usize;
        for child_ids in children_set.chunks(2) {
            let parent_ids: Vec<usize> = (0..self.tournament_selected.len())
                .into_iter()
                .choose_multiple(&mut self.rng, 2);


            let crossover_prob = rand::random::<f32>();
            if crossover_prob <= self.params.crossover_rate {
                match self.params.crossover_type {
                    0 => crossover_algos::single_point_crossover(self,
                                                                 &mut new_population,
                                                                 child_ids[0],
                                                                 child_ids[1],
                                                                 parent_ids[0],
                                                                 parent_ids[1]),
                    1 => crossover_algos::multi_point_crossover(self,
                                                                &mut new_population,
                                                                child_ids[0],
                                                                child_ids[1],
                                                                parent_ids[0],
                                                                parent_ids[1]),
                    2 => crossover_algos::uniform_crossover(self,
                                                            &mut new_population,
                                                            child_ids[0],
                                                            child_ids[1],
                                                            parent_ids[0],
                                                            parent_ids[1]),
                    3 => crossover_algos::no_crossover(self,
                                                       &mut new_population,
                                                       child_ids[0],
                                                       child_ids[1],
                                                       parent_ids[0],
                                                       parent_ids[1]),
                    _ => panic!("not implemented crossover tpye")
                }
            } else {
                //     no crossover, just copy parents
                new_population[child_ids[0]] = self.population[parent_ids[0]].clone();
                new_population[child_ids[1]] = self.population[parent_ids[1]].clone();
            }
        }
        self.population = new_population;
    }
}

