use std::fmt::{Display, Formatter};
use usize;
use rand::Rng;
use ndarray::prelude::*;
use crate::utils::boolean_functions as bf;
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
        match self.function_id {
            0 => { write!(f, "Func: AND, ")?; },
            1 => { write!(f, "Func: OR, ")?; },
            2 => { write!(f, "Func: NAND, ")?; },
            3 => { write!(f, "Func: NOR, ")?; },
            _ => panic!(),
        };
        return writeln!(f, "Connections: ({}, {}), ", self.connection0, self.connection1);
    }
}

impl Node {
    pub fn new(position: usize,
               nbr_inputs: usize,
               graph_width: usize,
               node_type: NodeType) -> Self {
        let function_id = rand::thread_rng().gen_range(0..=3) as usize;
        let connection0: usize;
        let connection1: usize;

        match node_type {
            NodeType::InputNode => {
                connection0 = usize::MAX;
                connection1 = usize::MAX;
            },
            NodeType::ComputationalNode => {
                connection0 = rand::thread_rng().gen_range(0..=position - 1);
                connection1 = rand::thread_rng().gen_range(0..=position - 1);
            },
            NodeType::OutputNode => {
                connection0 = rand::thread_rng().gen_range(0..=nbr_inputs + graph_width - 1);
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

    pub fn execute(&self, conn1_value: &ArrayView1<bool>, conn2_value: &ArrayView1<bool>) -> Array1<bool> {
        assert!(self.node_type != NodeType::InputNode);

        match self.function_id {
            0 => bf::and(conn1_value, conn2_value),
            1 => bf::or(conn1_value, conn2_value),
            2 => bf::nand(conn1_value, conn2_value),
            3 => bf::nor(conn1_value, conn2_value),
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
        self.function_id = gen_random_number_for_node(self.function_id, 4);
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
