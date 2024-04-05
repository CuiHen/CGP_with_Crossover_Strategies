use float_eq::float_eq;

pub fn add(con1: &Vec<f32>, con2: &Vec<f32>) -> Vec<f32> {
    return con1.iter().zip(con2.iter()).map(|(&a, &b)| a + b).collect();
}

pub fn subtract(con1: &Vec<f32>, con2: &Vec<f32>) -> Vec<f32> {
    return con1.iter().zip(con2.iter()).map(|(&a, &b)| a - b).collect();
}

pub fn mul(con1: &Vec<f32>, con2: &Vec<f32>) -> Vec<f32> {
    return con1.iter().zip(con2.iter()).map(|(&a, &b)| a * b).collect();
}

pub fn div(con1: &Vec<f32>, con2: &Vec<f32>) -> Vec<f32> {
    return con1.iter().zip(con2.iter()).map(|(&a, &b)| {
        if float_eq!(b, 0.0, abs <= 0.000_01) {
            return 1.;
        } else {
            return a / b;
        }
    }).collect();
}

pub fn sin(con1: &Vec<f32>) -> Vec<f32> {
    return con1.iter().map(|&x| x.sin()).collect();
}

pub fn cos(con1: &Vec<f32>) -> Vec<f32> {
    return con1.iter().map(|&x| x.cos()).collect();
}

pub fn ln(con1: &Vec<f32>) -> Vec<f32> {
    return con1.iter().map(|&x| {
        if float_eq!(x, 0.0, abs <= 0.000_01) {
            return 1.;
        } else {
            return x.abs().ln();
        }
    }
    ).collect();
}

pub fn exp(con1: &Vec<f32>) -> Vec<f32> {
    return con1.iter().map(|&x| x.exp()).collect();
}


// +, - *, /, sin, cos, ln(|n|), e^n