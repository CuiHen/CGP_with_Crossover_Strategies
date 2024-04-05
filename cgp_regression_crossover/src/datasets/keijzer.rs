use crate::utils::utility_funcs::float_loop;

fn make_label(inputs: &Vec<Vec<f32>>) -> Vec<f32> {
    let mut labels: Vec<f32> = vec![];
    for d in inputs {
        labels.push(d.iter().map(|x| 1. / x).sum::<f32>());
    }

    return labels;
}


pub fn get_dataset() -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut data = vec![];

    for x in float_loop(1., 50., 1.) {
        let mut elem: Vec<f32> = vec![];
        elem.push(x);

        data.push(elem);
    }

    let labels = make_label(&data);

    return (data, labels);
}

pub fn get_eval_dataset() -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut data = vec![];

    for x in float_loop(1., 120., 1.) {
        let mut elem: Vec<f32> = vec![];
        elem.push(x);

        data.push(elem);
    }


    let labels = make_label(&data);

    return (data, labels);
}


