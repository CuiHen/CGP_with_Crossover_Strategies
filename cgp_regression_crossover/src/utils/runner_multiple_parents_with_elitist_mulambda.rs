use std::fmt::{Display, Formatter};
use rand;
use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use itertools::Itertools;
use crate::global_params::CgpParameters as g_params;
use crate::utils::utility_funcs;
use crate::utils::utility_funcs::{get_argmin, get_argmins_of_value, vect_difference};
use crate::utils::crossover::crossover_algos;

#[cfg(feature = "standard")]
use crate::standard_cgp::chromosome::Chromosome;
#[cfg(feature = "ereorder")]
use crate::reorder::chromosome_reorder_equidistant::Chromosome;

// ID's: Begin with population, afterwards elitsts.
// Total number of population: population.len + elitsits.len
// Andere Idee:
// größere Population: Population <- population + elitisten
// speichere elitist id

pub struct Runner {
    pub params: g_params,
    data: Vec<Vec<f32>>,
    label: Vec<f32>,
    eval_data: Vec<Vec<f32>>,
    eval_label: Vec<f32>,
    pub population: Vec<Chromosome>,
    pub fitness_vals_sorted: Vec<f32>,
    // check for correctness, must include elitists too
    pub fitness_vals: Vec<f32>,
    pub rng: ThreadRng,
    pub elitist_ids: Vec<usize>,

    pub child_ids: Vec<usize>,
    // check for correctness, must include elitists too
    pub selected_parents_ids: Vec<usize>,
}

impl Display for Runner {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Fitnesses: {:?}", self.fitness_vals)
    }
}

impl Runner {
    pub fn new(params: g_params,
               data: Vec<Vec<f32>>,
               label: Vec<f32>,
               eval_data: Vec<Vec<f32>>,
               eval_label: Vec<f32>, ) -> Self {
        let mut rng = rand::thread_rng();

        let data = utility_funcs::transpose(data);
        let eval_data = utility_funcs::transpose(eval_data);

        let mut population: Vec<Chromosome> = Vec::with_capacity(params.population_size + params.elitism_number);
        let mut fitness_vals: Vec<f32> = Vec::with_capacity(params.population_size + params.elitism_number);

        for _ in 0..(params.population_size + params.elitism_number) {
            let mut chromosome = Chromosome::new(params.clone());
            let mut fitness = chromosome.evaluate(&data, &label);

            if fitness.is_nan() {
                fitness = f32::MAX;
            }

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
            eval_data,
            eval_label,
            population,
            fitness_vals,
            fitness_vals_sorted,
            rng,
            elitist_ids,
            selected_parents_ids: vec![],
            child_ids
        }
    }

    pub fn learn_step(&mut self, i: usize) {
        self.get_child_ids();

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
        // elitists should not be reordered as they did not change
        let reorder_set: Vec<usize> = (0..(self.params.population_size + self.params.elitism_number)).collect();
        let reorder_set = vect_difference(&reorder_set, &self.elitist_ids);

        for id in reorder_set {
            self.population[id].reorder();
        }
    }



    fn mutate_chromosomes(&mut self) {
        // mutate new chromosomes; do not mutate elitists
        for id in &self.child_ids {
            self.population[*id].mutate_single();
        }
    }

    fn eval_chromosomes(&mut self) {
        // for id in eval_set {
        for id in &self.child_ids {
            let mut fitness: f32 = self.population[*id].evaluate(&self.data, &self.label);

            if fitness.is_nan() {
                fitness = f32::MAX;
            } else if fitness.is_infinite() {
                fitness = f32::MAX;
            }

            self.fitness_vals[*id] = fitness;
        }

        let mut best_fitnesses_sorted = self.fitness_vals.clone();
        best_fitnesses_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        self.fitness_vals_sorted = best_fitnesses_sorted;
    }


    fn get_elitists(&mut self) {
        // Get mu - many best fitness vals
        let mut sorted_fitness_vals = self.fitness_vals_sorted.clone();
        // remove duplicates
        sorted_fitness_vals.dedup();

        let mut new_parent_ids: Vec<usize> = Vec::with_capacity(self.params.elitism_number);
        for current_best_fitness_val in sorted_fitness_vals {
            let mut parent_candidate_ids: Vec<usize> = Vec::with_capacity(self.params.population_size);

            get_argmins_of_value(&self.fitness_vals,
                                 &mut parent_candidate_ids,
                                 current_best_fitness_val);


            let remaining_new_parent_spaces = self.params.elitism_number - new_parent_ids.len();
            if parent_candidate_ids.len() <= remaining_new_parent_spaces {
                // if enough space left, extend all parent candidates
                new_parent_ids.extend(parent_candidate_ids);
            } else {
                //     case: more candidates than parent spaces left
                //     remove parents from the previous generation until either all parents removed
                //     or parent_candidates.len can fill remaining spaces

                // remove parent ids until either no parent ids are left or the candidate list fits
                // into the remaining new parent set
                for old_parent_id in &self.elitist_ids {
                    // if the old parent id is in candidate list
                    if parent_candidate_ids.contains(old_parent_id) {
                        // get index of parent in the candidate list
                        let index = parent_candidate_ids
                            .iter()
                            .position(|x| *x == *old_parent_id)
                            .unwrap();
                        // remove in O(1)
                        parent_candidate_ids.swap_remove(index);
                        // if enough parents are removed, break
                        if parent_candidate_ids.len() <= remaining_new_parent_spaces {
                            break;
                        }
                    }
                }

                parent_candidate_ids.truncate(self.params.elitism_number - new_parent_ids.len());
                new_parent_ids.extend(parent_candidate_ids);

                if new_parent_ids.len() == self.params.elitism_number {
                    break;
                }
            }
        }
        assert_eq!(self.elitist_ids.len(), new_parent_ids.len());
        self.elitist_ids = new_parent_ids;
    }
    pub fn get_best_fitness(&self) -> f32 {
        return self.fitness_vals_sorted[0];
    }

    pub fn get_test_fitness(&mut self) -> f32 {
        let mut best_fitness = f32::MAX;

        for individual in &mut self.population {
            let fitness = individual.evaluate(&self.eval_data, &self.eval_label);

            if !fitness.is_nan() && fitness < best_fitness {
                best_fitness = fitness;
            }
        }
        return best_fitness;
    }

    pub fn get_elitism_fitness(&self) -> Vec<f32> {
        let mut results: Vec<f32> = Vec::with_capacity(self.params.elitism_number);
        for id in &self.elitist_ids {
            results.push(self.fitness_vals[*id]);
        }
        return results;
    }

    pub fn get_parent(&self) -> Chromosome {
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

            let parent_ids: Vec<usize> = self.elitist_ids
                .choose_multiple(&mut self.rng, 2)
                .map(|x| *x)
                .collect();


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

