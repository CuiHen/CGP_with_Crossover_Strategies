use float_eq::float_eq;
use std::collections::HashSet;
use rand::distributions::{Distribution, Uniform};

pub fn get_argmins_of_value(vecs: &Vec<f32>, result_vec: &mut Vec<usize>, comp_value: f32) {
    vecs.iter()
        .enumerate()
        .for_each(|(i, v)| {
            if float_eq!(*v, comp_value, abs <= 0.000_1) {
                result_vec.push(i);
            }
        });
}

pub fn get_argmin(nets: &Vec<f32>) -> usize {
    nets.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .unwrap()
}

pub fn get_min(nets: &Vec<f32>) -> f32 {
    *nets.into_iter()
        .min_by(|a, b| a.partial_cmp(b)
            .unwrap())
        .unwrap()
}


pub fn vect_difference(v1: &Vec<usize>, v2: &Vec<usize>) -> Vec<usize> {
    let s1: HashSet<usize, nohash_hasher::BuildNoHashHasher<usize>> = v1.iter().cloned().collect();
    let s2: HashSet<usize, nohash_hasher::BuildNoHashHasher<usize>> = v2.iter().cloned().collect();
    (&s1 - &s2).iter().cloned().collect()
}



pub fn gen_random_number_for_node(excluded: usize, upper_range: usize) -> usize {
    if upper_range <= 1 {
        return 0;
    }

    let mut between = Uniform::from(0..=upper_range - 1);
    let mut rng = rand::thread_rng();

    loop {
        let rand_nbr: usize = between.sample(&mut rng);
        if rand_nbr != excluded {
            return rand_nbr;
        }
    }
}