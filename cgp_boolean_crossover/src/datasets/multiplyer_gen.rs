
pub fn print_dataset(n: usize) {
    let int_max = i32::from_str_radix(&"1".repeat(n), 2).unwrap() +1;
    // println!("itn max: {}", int_max);
    println!("Data:");

    for a in 0..int_max {
        let a_binary =  String::from(format!("{a:b}"));
        let a_binary = format!("{:0>n$}", a_binary);

        for b in 0..int_max {


            let b_binary =  String::from(format!("{b:b}"));
            let b_binary = format!("{:0>n$}", b_binary);

            let res: Vec<char> = format!("{}{}", a_binary, b_binary)
                .chars()
                .collect();
            let res: Vec<bool> = res.iter().map(|v| {if *v == '1' {true} else {false}}).collect();
            println!("{:?},", res);
        }
    }

    println!("\nLabel:");
    for a in 0..int_max {
        for b in 0..int_max {
            let res = a * b;
            let res =  String::from(format!("{res:b}"));
            let pad_size = 2 * n;
            let res = format!("{:0>pad_size$}", res);
            let res: Vec<char> = res.chars().collect();
            let res: Vec<bool> = res.iter().map(|v| {if *v == '1' {true} else {false}}).collect();

            println!("{:?},", res);

        }
    }

}

