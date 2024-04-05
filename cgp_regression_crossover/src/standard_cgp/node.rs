use std::fmt::{Display, Formatter};
use rand::Rng;
use ndarray::prelude::*;
// use crate::utils::boolean_functions as bf;
use crate::utils::symbolic_regression_functions as function_set;
use crate::utils::node_type::NodeType;
use crate::utils::utility_funcs::gen_random_number_for_node;


#[derive(Clone)]
pub struct Node {
    pub position: usize,
    pub node_type: NodeType,
    pub nbr_inputs: usize,
    pub graph_width: usize,
    pub function_id: usize,
    pub connection0: usize,
    pub connection1: usize,
}

impl Display for Node {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node Pos: {}, ", self.position)?;
        write!(f, "Node Type: {}, ", self.node_type)?;
        write!(f, "Function ID: {}, ", self.function_id)?;
        return writeln!(f, "Connections: ({}, {}), ", self.connection0, self.connection1);
    }
}

impl Node {
    pub fn new(position: usize,
               nbr_inputs: usize,
               graph_width: usize,
               node_type: NodeType) -> Self {
        let function_id: usize = rand::thread_rng().gen_range(0..=7);
        let connection0: usize;
        let connection1: usize;

        match node_type {
            NodeType::InputNode => {
                connection0 = usize::MAX;
                connection1 = usize::MAX;
            },
            NodeType::ComputationalNode => {
                connection0 = rand::thread_rng().gen_range(0..position);
                connection1 = rand::thread_rng().gen_range(0..position);
            },
            NodeType::OutputNode => {
                connection0 = rand::thread_rng().gen_range(0..nbr_inputs + graph_width);
                connection1 = usize::MAX;
            },
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

    pub fn execute(&self, conn1_value: &Vec<f32>, conn2_value: Option<&Vec<f32>>) -> Vec<f32> {
        assert!(self.node_type != NodeType::InputNode);

        match self.function_id {
            0 => function_set::add(conn1_value, conn2_value.unwrap()),
            1 => function_set::subtract(conn1_value, conn2_value.unwrap()),
            2 => function_set::mul(conn1_value, conn2_value.unwrap()),
            3 => function_set::div(conn1_value, conn2_value.unwrap()),
            4 => function_set::sin(conn1_value),
            5 => function_set::cos(conn1_value),
            6 => function_set::ln(conn1_value),
            7 => function_set::exp(conn1_value),
            _ => panic!("wrong function id: {}", self.function_id),
        }
    }

    pub fn mutate(&mut self) {
        assert!(self.node_type != NodeType::InputNode);

        match self.node_type {
            NodeType::OutputNode => self.mutate_output_node(),
            NodeType::ComputationalNode => self.mutate_computational_node(),
            _ => { panic!("Trying to mutate input node") }
        }
    }

    fn mutate_connection(connection: &mut usize, upper_range: usize) {
        *connection = gen_random_number_for_node(*connection,
                                                 upper_range);

    }

    fn mutate_function(&mut self) {
        self.function_id = gen_random_number_for_node(self.function_id, 8);
    }

    fn mutate_output_node(&mut self) {
        Node::mutate_connection(&mut self.connection0,
                                self.graph_width + self.nbr_inputs);

        assert!(self.connection0 < self.position);
    }

    fn mutate_computational_node(&mut self) {
        let rand_nbr = rand::thread_rng().gen_range(0..=2);
        match rand_nbr {
            0 => Node::mutate_connection(&mut self.connection0,
                                         self.position),

            1 => Node::mutate_connection(&mut self.connection1,
                                         self.position),

            2 => self.mutate_function(),

            _ => { panic!("Mutation: output node something wrong") }
        };

        assert!(self.connection0 < self.position);
        assert!(self.connection1 < self.position);
    }
}
