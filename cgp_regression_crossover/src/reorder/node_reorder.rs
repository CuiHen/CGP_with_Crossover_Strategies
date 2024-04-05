use std::fmt::{Display, Formatter};
use rand::Rng;
use ndarray::prelude::*;
use crate::utils::symbolic_regression_functions as function_set;
use crate::utils::node_type::NodeType;
use crate::utils::utility_funcs::gen_random_number_for_node;

#[derive(Clone)]
pub struct NodeReorder {
    pub position: usize,
    pub node_type: NodeType,
    pub nbr_inputs: usize,
    pub graph_width: usize,
    pub function_id: usize,
    pub connection0: usize,
    pub connection1: usize,
}

impl Display for NodeReorder {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node Pos: {}, ", self.position)?;
        write!(f, "Node Type: {}, ", self.node_type)?;
        match self.function_id {
            0 => { write!(f, "Func: AND, ")?; }
            1 => { write!(f, "Func: OR, ")?; }
            2 => { write!(f, "Func: NAND, ")?; }
            3 => { write!(f, "Func: NOR, ")?; }
            _ => panic!(),
        };
        return writeln!(f, "Connections: ({}, {}), ", self.connection0, self.connection1);
    }
}

impl NodeReorder {
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
            }
            NodeType::ComputationalNode => {
                connection0 = rand::thread_rng().gen_range(0..=position - 1);
                connection1 = rand::thread_rng().gen_range(0..=position - 1);
            }
            NodeType::OutputNode => {
                connection0 = rand::thread_rng().gen_range(0..=nbr_inputs + graph_width - 1);
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
        NodeReorder::mutate_connection(&mut self.connection0,
                                       self.graph_width + self.nbr_inputs);

        assert!(self.connection0 < self.position);
    }

    fn mutate_computational_node(&mut self) {
        let rand_nbr = rand::thread_rng().gen_range(0..=2);
        match rand_nbr {
            0 => NodeReorder::mutate_connection(&mut self.connection0,
                                                self.position),

            1 => NodeReorder::mutate_connection(&mut self.connection1,
                                                self.position),

            2 => self.mutate_function(),

            _ => { panic!("Mutation: output node something wrong") }
        };

        assert!(self.connection0 < self.position, "what was mutatet?: {}", rand_nbr);
        assert!(self.connection1 < self.position, "what was mutatet?: {}", rand_nbr);
    }

    pub fn set_new_position(&mut self, new_pos: usize, mutate_new_connections: bool) {
        if mutate_new_connections {
            if self.connection0 >= new_pos {
                NodeReorder::mutate_connection(&mut self.connection0,
                                               new_pos - 1);
            }
            if self.connection1 >= new_pos {
                NodeReorder::mutate_connection(&mut self.connection1,
                                               new_pos - 1);
            }
        }
        self.position = new_pos;
    }

    pub fn set_connection0(&mut self, new_con: usize) {
        self.connection0 = new_con;
    }

    pub fn set_connection1(&mut self, new_con: usize) {
        self.connection1 = new_con;
    }
}

