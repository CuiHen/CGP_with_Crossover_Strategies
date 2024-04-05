
pub fn fitness_regression(prediction: &Vec<f32>, label: &Vec<f32>) -> f32 {
    assert_eq!(prediction.len(), label.len());
    let mut fitness: f32 = 0.;
    prediction.iter().zip(label.iter()).for_each(|(x, y)| fitness +=  (x - y).abs() );

    fitness = fitness / (prediction.len() as f32);

    return fitness;


}