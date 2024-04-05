use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Formatter};
use nohash_hasher::BuildNoHashHasher;
use petgraph::algo::toposort;
use petgraph::graph::NodeIndex;
use petgraph::prelude::StableGraph;
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use crate::global_params::CgpParameters as g_params;
use crate::dag::node_dag::NodeDAG;
use crate::utils::node_type::NodeType;
use crate::utils::fitness_metrics;
use crate::utils::cycle_checker::CGPEdges;
use crate::utils::utility_funcs;


#[derive(Clone)]
pub struct Chromosome {
    pub params: g_params,
    pub nodes_grid: Vec<NodeDAG>,
    pub output_node_ids: Vec<usize>,
    pub active_nodes: Option<Vec<usize>>,
    pub cgp_edges: CGPEdges,
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
        let mut nodes_grid: Vec<NodeDAG> = Vec::with_capacity(params.nbr_inputs + params.nbr_computational_nodes + params.nbr_outputs);
        let mut output_node_ids: Vec<usize> = Vec::with_capacity(params.nbr_outputs);

        let mut cgp_edges = CGPEdges::new(params.nbr_inputs + params.nbr_computational_nodes);

        // input nodes
        for position in 0..params.nbr_inputs {
            nodes_grid.push(NodeDAG::new(position,
                                         params.nbr_inputs,
                                         params.nbr_computational_nodes,
                                         NodeType::InputNode,
                                         &mut cgp_edges,
            ));
        }
        // computational nodes
        for position in params.nbr_inputs..(params.nbr_inputs + params.nbr_computational_nodes) {
            nodes_grid.push(NodeDAG::new(position,
                                         params.nbr_inputs,
                                         params.nbr_computational_nodes,
                                         NodeType::ComputationalNode,
                                         &mut cgp_edges,
            ));
        }
        // output nodes
        for position in (params.nbr_inputs + params.nbr_computational_nodes)
            ..
            (params.nbr_inputs + params.nbr_computational_nodes + params.nbr_outputs) {
            nodes_grid.push(NodeDAG::new(position,
                                         params.nbr_inputs,
                                         params.nbr_computational_nodes,
                                         NodeType::OutputNode,
                                         &mut cgp_edges,
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
            cgp_edges,
        }
    }


    pub fn evaluate(&mut self, inputs: &Vec<Vec<f32>>, labels: &Vec<f32>) -> f32 {
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
            let current_node: &NodeDAG = &self.nodes_grid[*node_id];

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
                    let con0 = current_node.connection0;
                    let prev_output0 = outputs.get(&con0).unwrap();

                    let calculated_result: Vec<f32>;
                    if current_node.function_id <= 3 {  // case: two inputs needed
                        let con1 = current_node.connection1;
                        let prev_output1 = outputs.get(&con1).unwrap();

                        calculated_result = current_node.execute(&prev_output0, Some(&prev_output1));
                    } else {  // case: only one input needed
                        calculated_result = current_node.execute(&prev_output0, None);
                    }
                    outputs.insert(*node_id, calculated_result);
                }
            }
        }

        let output_start_id = self.params.nbr_inputs + self.params.nbr_computational_nodes;
        // let output_end_id = self.params.nbr_inputs + self.params.nbr_computational_nodes + self.params.nbr_outputs;
        let outs: &Vec<f32> = outputs.get(&output_start_id).unwrap();
        // println!("{:?}", outs);
        assert!(self.nodes_grid[output_start_id].node_type == NodeType::OutputNode);

        let fitness = fitness_metrics::fitness_regression(&outs, &labels);


        return fitness;
    }

    pub fn get_active_nodes_id(&mut self) {
        // let mut active: HashSet<usize, BuildNoHashHasher<usize>> = HashSet::with_capacity_and_hasher(
        //     self.params.nbr_inputs + self.params.nbr_computational_nodes + self.params.nbr_outputs,
        //     BuildNoHashHasher::default(),
        // );
        //
        // let mut to_visit: Vec<usize> = Vec::with_capacity(self.params.nbr_inputs + self.params.nbr_computational_nodes + self.params.nbr_outputs);
        //
        // // init new graph with its nodes
        // let mut graph = StableGraph::<usize, ()>::new();
        //
        // let mut nodes: Vec<NodeIndex> = Vec::with_capacity(self.params.nbr_inputs + self.params.nbr_computational_nodes + self.params.nbr_outputs);
        // for i in 0..(self.params.nbr_inputs + self.params.nbr_computational_nodes + self.params.nbr_outputs) {
        //     nodes.push(graph.add_node(i));
        // }
        //
        // // insert all output nodes, as they are always active
        // for output_node_id in &self.output_node_ids {
        //     active.insert(*output_node_id);
        //     to_visit.push(*output_node_id);
        // }
        let mut active: HashSet<usize, nohash_hasher::BuildNoHashHasher<usize>> = HashSet::default();
        active.reserve(self.params.nbr_inputs + self.params.nbr_computational_nodes + self.params.nbr_outputs);

        let mut to_visit: Vec<usize> = vec![];
        to_visit.reserve(self.params.nbr_inputs + self.params.nbr_computational_nodes + self.params.nbr_outputs);

        let mut graph = StableGraph::<usize, ()>::new();

        let mut nodes: Vec<NodeIndex> = vec![];
        nodes.reserve(self.params.nbr_inputs + self.params.nbr_computational_nodes + self.params.nbr_outputs);
        for i in 0..(self.params.nbr_inputs + self.params.nbr_computational_nodes + self.params.nbr_outputs) {
            nodes.push(graph.add_node(i));
        }
        for output_node_id in &self.output_node_ids {
            active.insert(*output_node_id);
            to_visit.push(*output_node_id);
        }

        // check for active
        while let Some(current_node_id) = to_visit.pop() {
            let current_node: &NodeDAG = &self.nodes_grid[current_node_id];

            match current_node.node_type {
                NodeType::InputNode => continue,

                NodeType::ComputationalNode => {
                    let connection0 = current_node.connection0;
                    graph.add_edge(nodes[connection0], nodes[current_node.position], ());

                    if !active.contains(&connection0) {
                        to_visit.push(connection0);
                        active.insert(connection0);
                    }

                    if current_node.function_id <= 3 {
                        let connection1 = current_node.connection1;
                        graph.add_edge(nodes[connection1], nodes[current_node.position], ());

                        if !active.contains(&connection1) {
                            to_visit.push(connection1);
                            active.insert(connection1);
                        }
                    }
                }

                NodeType::OutputNode => {
                    let connection0 = current_node.connection0;
                    graph.add_edge(nodes[connection0], nodes[current_node.position], ());

                    if !active.contains(&connection0) {
                        to_visit.push(connection0);
                        active.insert(connection0);
                    }
                }
            }
        }
        let inactive_nodes = (0..(self.params.nbr_inputs + self.params.nbr_computational_nodes)).collect();
        let inactive_nodes = utility_funcs::vect_difference(&inactive_nodes, &active.into_iter().collect());

        for i in inactive_nodes {
            graph.remove_node(nodes[i]);
        }


        let res = match toposort(&graph, None) {
            Ok(file) => file,
            Err(_) => {
                println!("Active: {:?}", self.active_nodes);
                println!("Output Nodes: {:?}", self.output_node_ids);

                println!("Active Nodes:");
                for i in self.active_nodes.clone().unwrap() {
                    println!("{}", self.nodes_grid[i]);
                }
                panic!()
            }
        };
        let res = res
            .into_iter()
            .map(|node| node.index())
            .collect::<Vec<usize>>();

        self.active_nodes = Some(res);
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

        let mut between = Uniform::from(start_id..end_id);
        let mut rng = rand::thread_rng();

        loop {
            let random_node_id = between.sample(&mut rng);
            self.nodes_grid[random_node_id].mutate(&mut self.cgp_edges);
            if self.active_nodes.as_ref().unwrap().contains(&random_node_id) {
                break;
            }
        }
    }

    pub fn mutate_prob(&mut self, prob: f32) {
        let mut start_id = self.params.nbr_inputs;
        if start_id == 1 {
            // Serious edge case: if start_id == 1; then only the first node can be mutated.
            // if its connection gets mutated, it can only mutate a connection to 0, because
            // the first node must have a connection to the input.
            // As the code currently forces a change of value, this will not terminate.
            start_id = 2;
        }
        let end_id = self.params.nbr_inputs + self.params.nbr_computational_nodes + self.params.nbr_outputs;
        for node_id in start_id..end_id {
            let random_prob: f32 = rand::thread_rng().gen::<f32>();
            if random_prob < prob {
                self.nodes_grid[node_id].mutate(&mut self.cgp_edges);
            };
        }
    }

    pub fn reorder(&mut self) {
        return;
    }

    fn get_dependency_graph(&self) -> StableGraph<usize, ()> {
        let mut graph = StableGraph::<usize, ()>::new();

        let mut node_indices: Vec<NodeIndex> = vec![];
        node_indices.reserve(self.params.nbr_inputs + self.params.nbr_computational_nodes + self.params.nbr_outputs);
        for i in 0..(self.params.nbr_inputs + self.params.nbr_computational_nodes + self.params.nbr_outputs) {
            node_indices.push(graph.add_node(i));
        }

        for current_node_id in 0..(self.params.nbr_inputs + self.params.nbr_computational_nodes + self.params.nbr_outputs) {
            let current_node: &NodeDAG = &self.nodes_grid[current_node_id];

            match current_node.node_type {
                NodeType::InputNode => continue,

                NodeType::ComputationalNode => {
                    let connection0 = current_node.connection0;
                    graph.add_edge(node_indices[connection0], node_indices[current_node.position], ());

                    let connection1 = current_node.connection1;
                    graph.add_edge(node_indices[connection1], node_indices[current_node.position], ());
                }

                NodeType::OutputNode => {
                    let connection0 = current_node.connection0;
                    graph.add_edge(node_indices[connection0], node_indices[current_node.position], ());

                }
            }
        }
        return graph;
    }

    /// sorts and repositions the genotype into a feed-forward graph for crossover
    pub fn unravel(&mut self) {
        self.create_new_genotype();

        self.renew_graph_dependencies();

        self.get_active_nodes_id();
    }

    fn get_new_locations_mapping(&mut self) -> HashMap<usize, usize, BuildNoHashHasher<usize>> {
        //     get the new positions of a node in a grid

        let mut topo_graph = self.get_dependency_graph();
        // Remove input and output nodes from the sorting process, as they should not change
        // position
        for node_id in 0..self.params.nbr_inputs {
            topo_graph.remove_node(NodeIndex::new(node_id));
        }
        for node_id in (self.params.nbr_computational_nodes + self.params.nbr_inputs)..(self.params.nbr_outputs + self.params.nbr_inputs + self.params.nbr_computational_nodes) {
            topo_graph.remove_node(NodeIndex::new(node_id));
        }

        // Sort nodes
        let sorted_node_indice = toposort(&topo_graph, None).unwrap();

        // convert NodeIndex to usize
        let sorted_node_indice = sorted_node_indice
            .into_iter()
            .map(|node| node.index())
            .collect::<Vec<usize>>();

        // Create a dictionary with new Locations
        // node index -> new location
        let mut new_location: HashMap<usize, usize, nohash_hasher::BuildNoHashHasher<usize>> = HashMap::default();
        new_location.reserve(self.params.nbr_outputs + self.params.nbr_inputs + self.params.nbr_computational_nodes);

        for (location, node_id) in sorted_node_indice.iter().enumerate() {
            // currently: location == ranking of the nodes
            // to get the new position:
            // -> shift the location by nbr_inputs.
            new_location.insert(*node_id, location + self.params.nbr_inputs);
        }

        return new_location;
    }

    fn create_new_genotype(&mut self) {
        let mut new_mapping: HashMap<usize, usize, BuildNoHashHasher<usize>> = self.get_new_locations_mapping();

        let mut new_nodes_grid: Vec<NodeDAG> = self.nodes_grid.clone();

        // insert old input and output nodes
        for node_id in 0..self.params.nbr_inputs {
            new_nodes_grid[node_id] = self.nodes_grid[node_id].clone();
        }
        for node_id in (self.params.nbr_computational_nodes + self.params.nbr_inputs)..(self.params.nbr_outputs + self.params.nbr_inputs + self.params.nbr_computational_nodes) {
            new_nodes_grid[node_id] = self.nodes_grid[node_id].clone();
        }

        // insert nodes into their new location
        for (node_id, new_location) in &new_mapping {
            new_nodes_grid[*new_location] = self.nodes_grid[*node_id].clone();
        }

        // update input and output nodes with same location for the mapping of nodes
        for node_id in 0..self.params.nbr_inputs {
            new_mapping.insert(node_id, node_id);
        }
        for node_id in (self.params.nbr_computational_nodes + self.params.nbr_inputs)..(self.params.nbr_outputs + self.params.nbr_inputs + self.params.nbr_computational_nodes) {
            new_mapping.insert(node_id, node_id);
        }

        // fix the connections for computational nodes
        for node_id in (self.params.nbr_inputs)..(self.params.nbr_computational_nodes + self.params.nbr_inputs) {
            assert!(*new_mapping.get(&new_nodes_grid[node_id].connection0).unwrap() < node_id);
            new_nodes_grid[node_id].connection0 = *new_mapping.get(&new_nodes_grid[node_id].connection0).unwrap();

            assert!(*new_mapping.get(&new_nodes_grid[node_id].connection1).unwrap() < node_id);
            new_nodes_grid[node_id].connection1 = *new_mapping.get(&new_nodes_grid[node_id].connection1).unwrap();

            new_nodes_grid[node_id].position = node_id;

            assert!(new_nodes_grid[node_id].connection0 < node_id);
            assert!(new_nodes_grid[node_id].connection1 < node_id);
        }

        // fix the connections for output nodes
        for node_id in (self.params.nbr_computational_nodes + self.params.nbr_inputs)..(self.params.nbr_outputs + self.params.nbr_inputs + self.params.nbr_computational_nodes) {
            assert!(*new_mapping.get(&new_nodes_grid[node_id].connection0).unwrap() < node_id);
            new_nodes_grid[node_id].connection0 = *new_mapping.get(&new_nodes_grid[node_id].connection0).unwrap();
        }

        self.nodes_grid = new_nodes_grid;
    }

    pub fn renew_graph_dependencies(&mut self) {
        let mut cgp_edges = CGPEdges::new(self.params.nbr_inputs + self.params.nbr_computational_nodes);

        for node_id in (self.params.nbr_inputs)..(self.params.nbr_computational_nodes + self.params.nbr_inputs) {
            cgp_edges.add_edge(node_id, self.nodes_grid[node_id].connection0);
            cgp_edges.add_edge(node_id, self.nodes_grid[node_id].connection1);
        }

        self.cgp_edges = cgp_edges;
    }
}