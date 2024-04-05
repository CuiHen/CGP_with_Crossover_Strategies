use std::fmt::{Display, Formatter};
use rand::Rng;
use rand::distributions::{Distribution, Uniform};
use crate::utils::symbolic_regression_functions as function_set;
use crate::utils::node_type::NodeType;
use crate::utils::cycle_checker::CGPEdges;

#[derive(Clone)]
pub struct NodeDAG {
    pub position: usize,
    pub node_type: NodeType,
    pub nbr_inputs: usize,
    pub graph_width: usize,
    pub function_id: usize,
    pub connection0: usize,
    pub connection1: usize,
}

impl Display for NodeDAG {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node Pos: {}, ", self.position)?;
        write!(f, "Node Type: {}, ", self.node_type)?;
        write!(f, "Function ID: {}, ", self.function_id)?;
        return writeln!(f, "Connections: ({}, {}), ", self.connection0, self.connection1);
    }
}

impl NodeDAG {
    pub fn new(position: usize,
               nbr_inputs: usize,
               graph_width: usize,
               node_type: NodeType,
               cgp_edges: &mut CGPEdges,
    ) -> Self {
        let function_id = rand::thread_rng().gen_range(0..=7) as usize;
        let connection0: usize;
        let connection1: usize;

        match node_type {
            NodeType::InputNode => {
                connection0 = usize::MAX;
                connection1 = usize::MAX;
            }
            NodeType::ComputationalNode => {
                connection0 = rand::thread_rng().gen_range(0..position);
                connection1 = rand::thread_rng().gen_range(0..position);
                cgp_edges.add_edge(position, connection0);
                cgp_edges.add_edge(position, connection1);
            }
            NodeType::OutputNode => {
                connection0 = rand::thread_rng().gen_range(0..nbr_inputs + graph_width);
                connection1 = usize::MAX;
            }
        }

        Self {
            position,
            node_type,
            nbr_inputs,
            graph_width,
            function_id,
            connection0,
            connection1,
        }
    }

    pub fn execute(&self, conn0_value: &Vec<f32>, conn1_value: Option<&Vec<f32>>) -> Vec<f32> {
        assert!(self.node_type != NodeType::InputNode);

        match self.function_id {
            0 => function_set::add(conn0_value, conn1_value.unwrap()),
            1 => function_set::subtract(conn0_value, conn1_value.unwrap()),
            2 => function_set::mul(conn0_value, conn1_value.unwrap()),
            3 => function_set::div(conn0_value, conn1_value.unwrap()),
            4 => function_set::sin(conn0_value),
            5 => function_set::cos(conn0_value),
            6 => function_set::ln(conn0_value),
            7 => function_set::exp(conn0_value),
            _ => panic!("wrong function id: {}", self.function_id),
        }
    }

    pub fn mutate(&mut self, cgp_edges: &mut CGPEdges) {
        match self.node_type {
            NodeType::OutputNode => self.mutate_output_node(),
            NodeType::ComputationalNode => self.mutate_computational_node(cgp_edges),
            _ => { panic!("Trying to mutate input node") }
        }
    }

    /// Upper Range excluded
    fn mutate_connection(connection: &mut usize, position: usize, upper_range: usize, cgp_edges: &mut CGPEdges) {
        let new_connection_id = gen_random_connection(*connection,
                                                      position,
                                                      upper_range,
                                                      cgp_edges, );
        cgp_edges.remove_edge(position, *connection);
        cgp_edges.add_edge(position, new_connection_id);
        *connection = new_connection_id;
    }

    fn mutate_function(&mut self) {
        self.function_id = gen_random_function_id(self.function_id, 8);
    }

    fn mutate_output_node(&mut self) {
        loop {
            let rand_nbr: usize = rand::thread_rng().gen_range(0..self.nbr_inputs + self.graph_width);

            if rand_nbr != self.connection0 {
                self.connection0 = rand_nbr;
                break;
            }
        }
    }

    fn mutate_computational_node(&mut self, cgp_edges: &mut CGPEdges) {
        let rand_nbr = rand::thread_rng().gen_range(0..=2);
        match rand_nbr {
            0 => NodeDAG::mutate_connection(&mut self.connection0,
                                            self.position,
                                            self.nbr_inputs + self.graph_width-1,
                                            cgp_edges),

            1 => NodeDAG::mutate_connection(&mut self.connection1,
                                            self.position,
                                            self.nbr_inputs + self.graph_width-1,
                                            cgp_edges),

            2 => self.mutate_function(),

            _ => { panic!("Mutation: output node something wrong") }
        };
    }
}


fn gen_random_function_id(excluded: usize, upper_range: usize) -> usize {
    loop {
        let rand_nbr: usize = rand::thread_rng().gen_range(0..upper_range);
        if rand_nbr != excluded {
            return rand_nbr;
        }
    }
}

fn gen_random_connection(previous_connection: usize, position: usize, upper_range: usize, cgp_edges: &mut CGPEdges) -> usize {
    let mut between = Uniform::from(0..upper_range);
    let mut rng = rand::thread_rng();

    loop {
        let rand_nbr: usize = between.sample(&mut rng);

        if (rand_nbr != previous_connection) && (rand_nbr != position) {
            if !cgp_edges.leads_to_cycle(position, rand_nbr) {
                return rand_nbr;
            }
        }
    }
}

