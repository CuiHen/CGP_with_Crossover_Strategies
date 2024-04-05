use crate::utils::utility_funcs::float_loop;


fn make_label(inputs: &Vec<Vec<f32>>) -> Vec<f32> {
    let mut labels: Vec<f32> = vec![];
    for d in inputs {
        labels.push(1. / (1. + d[0].powf(-4.)) + 1. / (1. + d[1].powf(-4.)));
    }

    return labels;
}


pub fn get_dataset() -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut data = vec![];

    for x in float_loop(-5., 5., 0.4) {
        for y in float_loop(-5., 5., 0.4) {
            let mut elem: Vec<f32> = vec![];
            elem.push(x);
            elem.push(y);

            data.push(elem);
        }
    }


    // x1 + (x2 * x3)
    let labels = make_label(&data);

    return (data, labels);
}

pub fn get_eval_dataset() -> (Vec<Vec<f32>>, Vec<f32>) {
    let (data, labels) = get_dataset();

    return (data, labels);
}

