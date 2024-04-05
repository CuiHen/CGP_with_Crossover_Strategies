use std::collections::HashSet;
use std::fmt::{Display, Formatter};
use ndarray::prelude::*;
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use crate::global_params::CgpParameters as g_params;
use crate::standard_cgp::node::Node;
use crate::utils::node_type::NodeType;
use crate::utils::fitness_metrics;

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


    pub fn evaluate(&mut self, inputs: &Array2<bool>, labels: &Array2<bool>) -> f32 {
        // let active_nodes = self.get_active_nodes_id();
        // self.active_nodes = Some(self.get_active_nodes_id());
        self.get_active_nodes_id();

        let output_size = inputs.slice(s![.., 0]).len();
        let nbr_nodes = self.params.nbr_inputs + self.params.nbr_computational_nodes + self.params.nbr_outputs;
        let nbr_nodes = nbr_nodes as usize;
        let mut outputs = Array::from_elem((nbr_nodes, output_size), true);

        for node_id in self.active_nodes.as_ref().unwrap() {
            // let node_id = *_node_id as usize;
            let current_node: &Node = &self.nodes_grid[*node_id];

            match current_node.node_type {
                NodeType::InputNode => {
                    // get the input from index node_id
                    let slice = inputs.slice(s![.., *node_id]);
                    // copy the slice to index node_id in outputs
                    let mut output_slice = outputs.slice_mut(s![*node_id, ..]);
                    output_slice.assign(&slice);
                }
                NodeType::OutputNode => {
                    let con1 = current_node.connection0;
                    let (mut output_slice, prev_output) = outputs.multi_slice_mut((s![*node_id, ..], s![con1, ..]));
                    output_slice.assign(&prev_output);
                }
                NodeType::ComputationalNode => {
                    let con1 = current_node.connection0;
                    let con2 = current_node.connection1;
                    let con1_slice = outputs.slice(s![con1, ..]);
                    let con2_slice = outputs.slice(s![con2, ..]);

                    let out = current_node.execute(&con1_slice, &con2_slice);
                    let mut output_slice = outputs.slice_mut(s![*node_id, ..]);
                    output_slice.assign(&out);
                }
            }
        }
        let output_start_id = self.params.nbr_inputs + self.params.nbr_computational_nodes;
        let output_end_id = self.params.nbr_inputs + self.params.nbr_computational_nodes + self.params.nbr_outputs;
        let outs = outputs.slice(s![output_start_id..output_end_id, ..]);
        let outs = outs.t();

        let fitness = fitness_metrics::fitness_boolean(&outs, &labels);

        return fitness;
    }

    pub fn get_active_nodes_id(&mut self) {
        let mut active: HashSet<usize, nohash_hasher::BuildNoHashHasher<usize>> = HashSet::default();
        active.reserve(self.params.nbr_inputs + self.params.nbr_computational_nodes + self.params.nbr_outputs);

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
                    let connection0 = current_node.connection1;
                    if !active.contains(&connection0) {
                        to_visit.push(connection0);
                        active.insert(connection0);
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
        let start_id = self.params.nbr_inputs;
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
