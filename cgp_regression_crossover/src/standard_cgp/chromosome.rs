use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Formatter};
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use crate::global_params::CgpParameters as g_params;
use crate::standard_cgp::node::Node;
use crate::utils::node_type::NodeType;
use crate::utils::fitness_metrics;
use nohash_hasher::BuildNoHashHasher;

#[derive(Clone)]
pub struct Chromosome {
    pub params: g_params,
    pub nodes_grid: Vec<Node>,
    pub output_node_ids: Vec<usize>,
    pub active_nodes: Option<Vec<usize>>,
}

impl Display for Chromosome {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        writeln!(f, "+++++++++++++++++ Chromosome +++++++++++")?;
        writeln!(f, "Nodes:")?;
        for node in &self.nodes_grid {
            write!(f, "{}", *node)?;
        }
        writeln!(f, "Active_nodes: {:?}", self.active_nodes)?;
        writeln!(f, "Output_nodes: {:?}", self.output_node_ids)
    }
}

impl Chromosome {
    pub fn new(params: g_params) -> Self {
        assert_eq!(params.nbr_outputs, 1);

        let mut nodes_grid: Vec<Node> = vec![];
        let mut output_node_ids: Vec<usize> = vec![];
        nodes_grid.reserve(params.nbr_inputs + params.nbr_computational_nodes + params.nbr_outputs);
        output_node_ids.reserve(params.nbr_outputs);

        // input nodes
        for position in 0..params.nbr_inputs {
            nodes_grid.push(Node::new(position,
                                      params.nbr_inputs,
                                      params.nbr_computational_nodes,
                                      NodeType::InputNode,
            ));
        }
        // computational nodes
        for position in params.nbr_inputs..(params.nbr_inputs + params.nbr_computational_nodes) {
            nodes_grid.push(Node::new(position,
                                      params.nbr_inputs,
                                      params.nbr_computational_nodes,
                                      NodeType::ComputationalNode,
            ));
        }
        // output nodes
        for position in (params.nbr_inputs + params.nbr_computational_nodes)
            ..
            (params.nbr_inputs + params.nbr_computational_nodes + params.nbr_outputs) {
            nodes_grid.push(Node::new(position,
                                      params.nbr_inputs,
                                      params.nbr_computational_nodes,
                                      NodeType::OutputNode,
            ));
        }

        // get position of output nodes
        for position in (params.nbr_inputs + params.nbr_computational_nodes)
            ..
            (params.nbr_inputs + params.nbr_computational_nodes + params.nbr_outputs) {
            output_node_ids.push(position);
        }

        Self {
            params,
            nodes_grid,
            output_node_ids,
            active_nodes: None,
        }
    }

    pub fn evaluate(&mut self, inputs: &Vec<Vec<f32>>, labels: &Vec<f32>) -> f32 {
        // let active_nodes = self.get_active_nodes_id();
        // self.active_nodes = Some(self.get_active_nodes_id());
        self.get_active_nodes_id();


        let mut outputs: HashMap<usize, Vec<f32>, BuildNoHashHasher<usize>> = HashMap::with_capacity_and_hasher(
            self.params.nbr_inputs + self.params.nbr_computational_nodes + self.params.nbr_outputs,
            BuildNoHashHasher::default(),
        );

        // iterate through each input and calculate for each new vector its output
        // as the inputs are transposed, the n-th element of the whole dataset is input
        // i.e. given a dataset with 3 datapoints per entry; and 5 entries.
        // then it will input the first datapoint of all 5 entries first. Then the second, etc.
        for node_id in self.active_nodes.as_ref().unwrap() {
            // println!("{:?}", input_slice);
            let current_node: &Node = &self.nodes_grid[*node_id];

            match current_node.node_type {
                NodeType::InputNode => {
                    outputs.insert(*node_id, inputs[*node_id].clone());
                }
                NodeType::OutputNode => {
                    let con1 = current_node.connection0;
                    let prev_output1 = outputs.get(&con1).unwrap();
                    outputs.insert(*node_id, prev_output1.clone());
                }
                NodeType::ComputationalNode => {
                    let con1 = current_node.connection0;
                    let prev_output1 = outputs.get(&con1).unwrap();

                    let calculated_result: Vec<f32>;
                    if current_node.function_id <= 3 {  // case: two inputs needed
                        let con2 = current_node.connection1;
                        let prev_output2 = outputs.get(&con2).unwrap();

                        calculated_result = current_node.execute(&prev_output1, Some(&prev_output2));
                    } else {  // case: only one input needed
                        calculated_result = current_node.execute(&prev_output1, None);
                    }
                    outputs.insert(*node_id, calculated_result);
                }
            }
        }

        let output_start_id = self.params.nbr_inputs + self.params.nbr_computational_nodes;
        let outs: &Vec<f32> = outputs.get(&output_start_id).unwrap();
        assert!(self.nodes_grid[output_start_id].node_type == NodeType::OutputNode);

        let fitness = fitness_metrics::fitness_regression(&outs, &labels);

        return fitness;
    }

    pub fn get_active_nodes_id(&mut self) {
        let mut active: HashSet<usize, BuildNoHashHasher<usize>> = HashSet::with_capacity_and_hasher(
            self.params.nbr_inputs + self.params.nbr_computational_nodes + self.params.nbr_outputs,
            BuildNoHashHasher::default(),
        );

        let mut to_visit: Vec<usize> = vec![];
        to_visit.reserve(self.params.nbr_inputs + self.params.nbr_computational_nodes + self.params.nbr_outputs);

        for output_node_id in &self.output_node_ids {
            active.insert(*output_node_id);
            to_visit.push(*output_node_id);
        }

        while let Some(current_node_id) = to_visit.pop() {
            let current_node: &Node = &self.nodes_grid[current_node_id];

            match current_node.node_type {
                NodeType::InputNode => continue,

                NodeType::ComputationalNode => {
                    let connection0 = current_node.connection0;
                    if !active.contains(&connection0) {
                        to_visit.push(connection0);
                        active.insert(connection0);
                    }
                    if current_node.function_id <= 3 {
                        // case: it needs two inputs instead of just one
                        let connection0 = current_node.connection1;
                        if !active.contains(&connection0) {
                            to_visit.push(connection0);
                            active.insert(connection0);
                        }
                    }
                }

                NodeType::OutputNode => {
                    let connection0 = current_node.connection0;
                    if !active.contains(&connection0) {
                        to_visit.push(connection0);
                        active.insert(connection0);
                    }
                }
            }
        }
        let mut active: Vec<usize> = active.into_iter().collect();
        active.sort_unstable();

        self.active_nodes = Some(active);
    }

    pub fn mutate_single(&mut self) {
        let mut start_id = self.params.nbr_inputs;
        if start_id == 1 {
            // Serious edge case: if start_id == 1; then only the first node can be mutated.
            // if its connection gets mutated, it can only mutate a connection to 0, because
            // the first node must have a connection to the input.
            // As the code currently forces a change of value, this will not terminate.
            start_id = 2;
        }
        let end_id = self.params.nbr_inputs + self.params.nbr_computational_nodes + self.params.nbr_outputs;

        let mut between = Uniform::from(start_id..=end_id - 1);
        let mut rng = rand::thread_rng();

        loop {
            let random_node_id = between.sample(&mut rng);
            self.nodes_grid[random_node_id].mutate();

            if self.active_nodes.as_ref().unwrap().contains(&random_node_id) {
                break;
            }
        }
    }

    pub fn reorder(&mut self) {
        return;
    }
}
