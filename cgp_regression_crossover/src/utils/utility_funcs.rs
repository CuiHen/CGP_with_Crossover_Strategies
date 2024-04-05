use float_eq::float_eq;
use std::collections::HashSet;
use rand::distributions::{Distribution, Uniform};

pub fn get_argmins_of_value(vecs: &Vec<f32>, res: &mut Vec<usize>, comp_value: f32) {
    vecs.iter()
        .enumerate()
        .for_each(|(i, v)| {
            if *v == comp_value {
            // if float_eq!(*v, comp_value, abs <= 0.000_01) {
                res.push(i);
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
    /// * upper_range is inclusive
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

pub fn transpose<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
    assert!(!v.is_empty());
    let len = v[0].len();
    let mut iters: Vec<_> = v.into_iter().map(|n| n.into_iter()).collect();
    (0..len)
        .map(|_| {
            iters
                .iter_mut()
                .map(|n| n.next().unwrap())
                .collect::<Vec<T>>()
        })
        .collect()
}

pub fn float_loop(start: f32, threshold: f32, step_size: f32) -> impl Iterator<Item=f32> {
    let threshold: f32 = threshold + 1.;
    std::iter::successors(Some(start), move |&prev| {
        let next = prev + step_size;
        (next < threshold).then_some(next)
    })
}