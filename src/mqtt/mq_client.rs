/**
 * mqtt客户端
 */
use crate::conf::index_conf;
use std::{
    io::{self, Read, Write},
    net::{SocketAddr, TcpStream},
    str::from_utf8,
    thread,
};

extern crate futures;
// extern crate futures_io;
// extern crate futures_mio;

// use futures::Future;
// use futures_mio::Loop;

pub fn create_client() {
    // let mut lp = Loop::new().unwrap();
    // let addr = "127.0.0.1:8080".parse::<SocketAddr>().unwrap();

    // let socket = lp.handle().tcp_connect(&addr);

    // let request = socket.and_then(|socket| {
    //     futures_io::write_all(socket, b"Hello!")
    // });

    // let response = request.and_then(|(socket, _)| {
    //     futures_io::read_to_end(socket, Vec::new())
    // });

    // let data = lp.run(response).unwrap();
    // println!("{}", String::from_utf8_lossy(&data));
    let port = index_conf::CONF_MAP.get("mqtt_port").unwrap();
    let client = TcpStream::connect(format!("localhost:{}", port)).unwrap();
    let mut guess = String::new();
    // response_handle(&client);
    loop {
        io::stdin()
            .read_line(&mut guess)
            .expect("Failed to read line");
        let result = guess.trim();
        send_msg(&client, result);
    }
}

fn send_msg(mut client: &TcpStream, msg: &str) {
    match client.write(msg.as_bytes()) {
        Ok(result) => {
            println!("result:{result}")
        },
        Err(e) => {
            println!("e:{e}")
        }
    };
}

fn response_handle(mut client: &TcpStream) {
    // let mut data = [0 as u8; 1024];
    let mut buffer = Vec::new();
    match client.read_to_end(&mut buffer) {
        Ok(_) => {
            if buffer.len() == 0 {
                return;
            }
            let text = from_utf8(&buffer).unwrap();
            println!("reply: {}, size:{}", text, buffer.len());
        }
        Err(e) => {
            println!("Failed to receive data: {}", e);
        }
    }
}
