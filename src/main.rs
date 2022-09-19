mod mqtt;
use mqtt::mq_server;

fn main() {
    let _a = "df";
    // const B: &str = "djxc";     // 常量
    // let c: i32 = "34".parse().expect("not number");
    println!("Hello, world123");
    println!("\t\t\t\n");
    println!("Hello, world");
    mq_server::create_server();
    greet_world();
    parse_str();
}

/**
 * rust中可以使用utf-8任意的字符
 */
fn greet_world() {
    let southern_germany = "Grüß Gott!";
    let chinese = "世界，你好";
    let english = "hello, world";
    let regions = [southern_germany, chinese, english];
    for region in regions {
        println!("{}", region)
    }
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
