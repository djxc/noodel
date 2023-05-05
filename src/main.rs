mod mqtt;
mod conf;
mod study;
mod vulkan_CV;

use std::net::TcpStream;
use std::{process::exit};

use mqtt::mq_server;
use mqtt::mq_client;
use study::base_study;

use crate::mqtt::mq_client::Client;
use crate::mqtt::mq_server::Server;
// use async_std::prelude::*;

// #[async_std::main]
// async
fn main() {
    // base_study::base_study();
    // base_study::noodel_version();
    parse_args();
}

/**
 * 解析参数
 * 1、args中第一个为当前可执行文件路径，其他的为用户输入的参数
 * 2、参数形式为 `参数名称=参数值`
 */
fn parse_args() {
    let mut index = 0;
    let args = std::env::args();
    if args.len() == 1 {
        println!("
        1、run as server: cargo run agent=server
        2、run as client: cargo run agent=client
        ");
        exit(-1);
    }

    for arg in args {
        println!("arg is {}", arg);
        if index > 0 {
            let arg_pars: Vec<_> = arg.split("=").collect();
            if arg_pars.len() < 2 {
                exit(-1);
            } else {
                let arg_type = arg_pars[0];
                let arg_value = arg_pars[1];
                if arg_type == "agent" {
                    match arg_value {
                        "server" => {
                            let mqtt_broker = mq_server::MqttBroker{};
                            mqtt_broker.start(8080);
                            // mq_server::create_server(6);
                        },
                        "client" => {
                            let tcp_stream = mq_client::connect_broker("localhost:8080");
                            let mut mqtt_client = mq_client::MqttClient{stream: tcp_stream};
                            // mqtt_client.send_msg("hello");
                            // mq_client::create_client(),
                        }
                        _ => println!("not find args")
                    }
                }
            }
            
        }
        index = index + 1;
    }
}



