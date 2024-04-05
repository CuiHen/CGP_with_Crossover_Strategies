use ndarray::{arr2, Array2};

pub fn get_dataset() -> (Array2<bool>, Array2<bool>) {
    let data = [
        [true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false],
        [false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false],
        [false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false],
        [false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false],
        [false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false],
        [false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false],
        [false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false],
        [false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false],
        [false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false],
        [false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false],
        [false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false],
        [false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false],
        [false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false],
        [false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false],
        [false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false],
        [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true], ];

    let labels = [
        [false, false, false, false],
        [false, false, false, true],
        [false, false, true, false],
        [false, false, true, true],
        [false, true, false, false],
        [false, true, false, true],
        [false, true, true, false],
        [false, true, true, true],
        [true, false, false, false],
        [true, false, false, true],
        [true, false, true, false],
        [true, false, true, true],
        [true, true, false, false],
        [true, true, false, true],
        [true, true, true, false],
        [true, true, true, true], ];

    let data = arr2(&data);
    let labels = arr2(&labels);

    return (data, labels);
}

