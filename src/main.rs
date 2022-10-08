mod mqtt;
mod conf;
use std::{process::exit};

use mqtt::mq_server;
use mqtt::mq_client;
// use async_std::prelude::*;

// #[async_std::main]
// async
fn main() {
    greet_world();
    parse_str();
    parse_args();
}

/**
 * 解析参数
 * 1、args中第一个为当前可执行文件路径，其他的为用户输入的参数
 * 2、参数形式为 `参数名称=参数值`
 */
fn parse_args() {
    let mut index = 0;
    for arg in std::env::args() {
        if index > 0 {
            let arg_pars: Vec<_> = arg.split("=").collect();
            if arg_pars.len() < 2 {
                exit(-1);
            } else {
                let arg_type = arg_pars[0];
                let arg_value = arg_pars[1];
                if arg_type == "agent" {
                    match arg_value {
                        "server" => mq_server::create_server(6),
                        "client" => mq_client::create_client(),
                        _ => println!("not find args")
                    }
                }
            }
            
        }
        index = index + 1;
    }
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
