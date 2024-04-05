use std::fmt::{Display, Formatter};

#[derive(Clone)]
pub struct CgpParameters {
    pub nbr_computational_nodes: usize,
    pub population_size: usize,
    pub mu: usize,
    pub lambda: usize,
    pub eval_after_iterations: usize,
    pub nbr_inputs: usize,
    pub nbr_outputs: usize,
    // pub mutation_type: usize,
    // pub mutation_rate: f32,
    pub crossover_type: usize,
    pub crossover_rate: f32,
    pub tournament_size: usize,
    pub elitism_number: usize,
    pub multi_point_n: usize,
    // pub unravel: usize,
    pub cgp_type: usize,
}

impl Default for CgpParameters {
    fn default() -> Self {
        CgpParameters {
            nbr_computational_nodes: 0,
            population_size: 0,
            mu: 1,
            lambda: 4,
            eval_after_iterations: 500,
            nbr_inputs: 0,
            nbr_outputs: 0,
            // mutation_type: 0,
            // mutation_rate: -1.0,
            crossover_type: 0,
            crossover_rate: -1.0,
            tournament_size: 0,
            elitism_number: 0,
            multi_point_n: 0,
            // unravel: 0,
            cgp_type: 0,
        }
    }
}

impl Display for CgpParameters {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "############ Parameters ############\n")?;
        write!(f, "graph_width: {}\n", self.nbr_computational_nodes)?;
        write!(f, "mu: {}\n", self.mu)?;
        write!(f, "lambda: {}\n", self.lambda)?;
        write!(f, "eval_after_iterations: {}\n", self.eval_after_iterations)?;
        write!(f, "nbr_inputs: {}\n", self.nbr_inputs)?;
        write!(f, "nbr_outputs: {}\n", self.nbr_outputs)?;
        // write!(f, "mutation_type: {}\n", self.mutation_type)?;
        // write!(f, "mutation_rate: {}\n", self.mutation_rate)?;
        write!(f, "crossover_type: {}\n", self.crossover_type)?;
        write!(f, "crossover_rate: {}\n", self.crossover_rate)?;
        write!(f, "multi_point_n: {}\n", self.multi_point_n)?;
        write!(f, "#########################\n")
    }
}