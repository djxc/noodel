mod mqtt;
mod util;
mod conf;
mod study;
mod DGIS;
mod vulkan_CV;

use std::net::TcpStream;
use std::{process::exit};

/// 如果引用的结构体实现了某个trait需要引用这个trait
use crate::mqtt::mq_server::Server;

use mqtt::mq_server;
use mqtt::mq_client;
use study::base_study;
use util::image_util;
use DGIS::SpatialEntity::DPoint;
use vulkan_CV::vulkan_demo;
use vulkan_CV::vulkano_demo;


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
    let args = std::env::args();
    if args.len() == 1 {
        println!("
        1、run as server: cargo run agent=server
        2、run as client: cargo run agent=client
        3、run as dGIS: cargo run agent=dGIS
        ");
        exit(-1);
    }

    for (index, arg) in args.enumerate() {
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
                        },
                        "dGIS" => {
                            // vulkan_demo::test_vulkan_window();
                            vulkan_demo::window_vulkan();
                            // vulkano_demo::vulkan_demo_test();
                        },
                        _ => println!("not find args")
                    }
                }
            }
            
        }
    }
}



