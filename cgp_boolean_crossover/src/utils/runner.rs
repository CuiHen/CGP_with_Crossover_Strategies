use std::fmt::{Display, Formatter};
use ndarray::Array2;
use rand::seq::SliceRandom;
use crate::global_params::CgpParameters as g_params;
use crate::utils::utility_funcs;

#[cfg(feature = "standard")]
use crate::standard_cgp::chromosome::Chromosome;
#[cfg(feature = "ereorder")]
use crate::reorder::chromosome_reorder_equidistant::Chromosome;


pub struct Runner {
    params: g_params,
    data: Array2<bool>,
    label: Array2<bool>,
    population: Vec<Chromosome>,
    best_fitness: f32,
    fitness_vals: Vec<f32>,
    parent_id: usize,
}

impl Display for Runner {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Parent: {}", self.population[self.parent_id])?;
        writeln!(f, "Fitness: {}", self.best_fitness)
    }
}

impl Runner {
    pub fn new(params: g_params,
               data: Array2<bool>,
               label: Array2<bool>) -> Self {
        let mut chromosomes: Vec<Chromosome> = Vec::with_capacity(params.mu + params.lambda);
        let mut fitness_vals: Vec<f32> = Vec::with_capacity(params.mu + params.lambda);

        for _ in 0..(params.mu + params.lambda) {
            let mut chromosome = Chromosome::new(params.clone());
            let fitness = chromosome.evaluate(&data, &label);
            fitness_vals.push(fitness);

            chromosomes.push(chromosome);
        }

        let best_fitness = utility_funcs::get_min(&fitness_vals);
        let parent_id = utility_funcs::get_argmin(&fitness_vals);

        Self {
            params,
            data,
            label,
            population: chromosomes,
            best_fitness,
            fitness_vals,
            parent_id,
        }
    }

    pub fn learn_step(&mut self, i: usize) {
        self.reorder();

        self.mutate_chromosomes();

        self.eval_chromosomes();

        self.new_parent_by_neutral_search();

    }

    fn reorder(&mut self) {
        // elitists should not be reordered as they did not change
        for id in 0..self.params.lambda + self.params.mu {
            if id == self.parent_id {
                continue;
            }
            self.population[id].reorder();
        }
    }


    fn new_parent_by_neutral_search(&mut self) {
        let mut min_keys: Vec<usize> = Vec::with_capacity(self.params.mu + self.params.lambda);

        utility_funcs::get_argmins_of_value(&self.fitness_vals, &mut min_keys, self.best_fitness);

        if min_keys.len() == 1 {
            self.parent_id = min_keys[0];
        } else {
            if min_keys.contains(&self.parent_id) {
                let index = min_keys.iter().position(|x| *x == self.parent_id).unwrap();
                min_keys.remove(index);
            }
            self.parent_id = *min_keys.choose(&mut rand::thread_rng()).unwrap();
        }
    }

    fn mutate_chromosomes(&mut self) {
        // mutate new chromosomes; do not mutate parent
        for i in 0..(self.params.mu + self.params.lambda) {
            if i == self.parent_id {
                continue;
            }

            self.population[i] = self.population[self.parent_id].clone();
            self.population[i].mutate_single();
        }
    }

    fn eval_chromosomes(&mut self) {
        for i in 0..(self.params.mu + self.params.lambda) {
            if i != self.parent_id {
                let fitness = self.population[i].evaluate(&self.data, &self.label);
                self.fitness_vals[i] = fitness;
            }
        }

        let best_fitness = utility_funcs::get_min(&self.fitness_vals);

        self.best_fitness = best_fitness;
    }

    pub fn get_best_fitness(&self) -> f32 {
        return self.best_fitness;
    }

    pub fn get_elitism_fitness(&self) -> Vec<f32> {
        return vec![self.best_fitness];
    }

    pub fn get_best_solution(&self) -> Chromosome {
        return self.population[self.parent_id].clone();
    }

}

