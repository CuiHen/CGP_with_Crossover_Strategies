use ndarray::{arr2, Array2};

pub fn get_dataset() -> (Array2<bool>, Array2<bool>) {
    let data = [
        [false, false, false],
        [false, false, true],
        [false, true, false],
        [false, true, true],
        [true, false, false],
        [true, false, true],
        [true, true, false],
        [true, true, true], ];

    let labels = [
        [true],
        [false],
        [false],
        [true],
        [false],
        [true],
        [true],
        [false]];

    let data = arr2(&data);
    let labels = arr2(&labels);

    return (data, labels);
}

