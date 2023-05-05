// 学习rust基础


pub fn base_study() {
    hello_world();
    let mut x: i32 = 6;
    print!("{x}");
    while x != 1 {
        if x % 2 == 0 {
            x = x / 2;
        } else {
            x = 3 * x + 1;
        }
        print!(" -> {x}");
    }
    
    array_test();
    tuple_demo();
    reference_demo(x);

    slice_demo();
    str_string_demo();
    let mut polygon = Polygon {area: 30.2, color: 10};
    polygon.show();
    let price = polygon.price();
    println!("price: {price}");
    let a: i8 = 32;
    let b: i16 = 16;
    let result = multiply(a.into(), b);
    println!("result: {result}");

    let matrix = [
        [101, 102, 103], // <-- the comment makes rustfmt add a newline
        [201, 202, 203],
        [301, 302, 303],
    ];

    println!("matrix:");
    pretty_print(&matrix);

    let transposed = transpose(matrix);
    println!("transposed:");
    pretty_print(&transposed);
    ownship_demo(&mut polygon);      // 变量作为函数参数也会进行所有权的转换，调用完函数，该变量不可使用。
    // 1、如果变量具有copy、clone特征则会复制一份。不会转移所有权
    // 2、将变量的值传递过去，而不传递变量。调用完之后会归还所有权
    println!("polygon1: {}", polygon.area);

    let polygon1 = Polygon {area: 30.2, color: 10};
    let polygon2 = Polygon {area: 30.2, color: 10};
    left_most(&polygon1, &polygon2);

}

/**
 * rust中可以使用utf-8任意的字符
 */
fn hello_world() {
    let southern_germany = "Grüß Gott!";
    let chinese = "世界，你好";
    let english = "hello, world";
    let regions = [southern_germany, chinese, english];
    for region in regions {
        println!("{}", region)
    }
}

pub fn noodel_version() {
    let version = "0.0.1";
    println!("noodel version: {}", version);
}

fn left_most<'a>(p1: &'a Polygon, p2: &'a Polygon) -> &'a Polygon {
    if p1.area < p2.area { p1 } else { p2 }
}

fn ownship_demo(polygon: &mut Polygon) {
    // rust中每个变量都有所有权，当所有权借出去之后则之前的变量不可使用。同一时刻仅有一个变量拥有所有权
    {
        let polygon1 = polygon;
        println!("area: {}", polygon1.area);
        polygon1.area = 23.2;
    }
    // println!("area: {}", polygon.area);
    let name = String::from("djxc");
    let name1 = name;
    println!("name: {}", name1);
}


fn transpose(matrix: [[i32; 3]; 3]) -> [[i32; 3]; 3] {
    let mut new_matrix: [[i32; 3]; 3] = matrix;
    for i in 0..3 {
        let sub_matrix = matrix[i];
        for j in 0..3 {
            new_matrix[j][i] = sub_matrix[j];
        }
    }
    return new_matrix;
}

fn pretty_print(matrix: &[[i32; 3]; 3]) {
    for i in matrix {
        for j in i {
            print!("{j},");
        }
        println!();
    }
}


fn multiply(x: i16, y: i16) -> i16{
    x * y
}

fn add_operate<T>(a: T, b: T) -> T {
    return a;
}

// rust中没有类，可以通过结构体构建一个对象，然后实现一些方法。
#[derive(Clone, Copy)]
struct Polygon {
    area: f32,
    color: i32,
}

// 为结构体增加方法
impl Polygon {
    fn price(&self) -> f32 {
        return self.area * 0.56;
    }

    fn show(&self) {
        println!("polygon: area->{}; color:{}", self.area, self.color)
    }
}


fn str_string_demo() {
    let s1:&str = "djxc";
    println!("s1: {s1}");
    let mut s2: String = String::from("djxc111");
    s2.push_str(" jamei");
    println!("s2: {s2}");
}

fn slice_demo() {
    let a: [i32; 10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0];
    println!("a:{a:?}");
    let s: &[i32] = &a[2..5];       // 切片之后所有权转移，不可修改内容
    println!("s:{s:?}");
}

fn reference_demo(mut x: i32) {
    // 引用,确定引用类型，根据引用可以修改指向的值。引用类似c的指针
    let ref_x: &mut i32 = &mut x;
    *ref_x = 20;
    println!("x -> {x}");

    // 悬挂指针,编译器会报错，编译不通过。
    let ref_x2: &mut i32;
    {
        let mut y: i32 = 30;
        ref_x2 = &mut y;
    }
    // println!("ref_x2 -> {ref_x2}");
}


fn array_test() {
    // 数组，定义数组类型以及长度
    let mut array_demo: [i32; 10] = [42; 10];
    array_demo[0] = 10;
    array_demo[1] = 9;
    println!("a:{:?}", array_demo);
}

fn tuple_demo() {
   // tuple类型
   let t: (i32, bool) = (7, true);
   println!("1st index: {}", t.0);
   println!("2nd index: {}", t.1);
}


/**
 * 解析字符串
 */
fn parse_str() {
    let penguin_data = "\
   common name,length (cm)
   Little penguin,33
   Yellow-eyed penguin,65
   Fiordland penguin,60
   Invalid,data
   ";
    let records = penguin_data.lines();
    for (i, record) in records.enumerate() {
        if i == 0 || record.trim().len() == 0 {
            continue;
        }
        let fields: Vec<_> = record.split(",").map(|f| f.trim()).collect();
        if cfg!(debug_assertions) {
            eprintln!("debug: {:?} -> {:?}", record, fields);
        }
        let name = fields[0];
        if let Ok(length) = fields[1].parse::<f32>() {
            println!("{}, {}cm", name, length);
        }
    }
}