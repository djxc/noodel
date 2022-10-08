use super::threadPool::ThreadPool;
use crate::conf::index_conf;
/**
 * 通过tcp启动服务器，实现mqtt协议
 */
use std::borrow::Cow;
use std::fs;
use std::sync::Arc;
use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
};
use std::{thread, time::Duration};

pub fn create_server(thead_num: usize) {
    // 堵塞请求 - 每秒处理请求为上限在90左右，每个请求用时为218ms    
    let port = index_conf::CONF_MAP.get("mqtt_port").unwrap();
    let listener = TcpListener::bind(format!("localhost:{}", port)).unwrap();
    let ls = Arc::new(listener);
    println!("server is running at :{}", port);
    let ls1 = ls.clone();
    // 需要创一个线程用来接收新客户端的连接
    thread::spawn(move || {
        let thread_pool = ThreadPool::new(thead_num);
        for stream in ls1.incoming() {
            let _stream = stream.unwrap();
            thread_pool.execute(|| {
                hander_connect(_stream);
            });
        }
    });

    // loop {
    //     let (stream, addr) = match ls.accept() {
    //         Ok((s, r)) => (s, r),
    //         Err(e) => {
    //             continue;
    //         }
    //     };
    //     hander_connect(stream);
    //     println!("new connect")
    // }
    
}

/**
 * 处理请求的handler
 * 通过tcpstream写入数据，进行返回
 * @param stream tcp管道流
 * @return none
 */
fn hander_connect(mut stream: TcpStream) {
    const OK_REPONSE_HEADER: &str = "HTTP/1.1 200 OK";
    // 定义byte数组，用来存放请求信息；通过stream的read读取stream中的请求到byte数组中
    let mut buffer = [0; 1024];
    stream.read(&mut buffer).unwrap();
    let request = String::from_utf8_lossy(&buffer[..]);
    parse_request(request);
    // let content = read_html_file("d:\\index.html");
    thread::sleep(Duration::from_millis(10));
    let content = "hello";
    let response_str = format!(
        "{}\r\nContent-Length:{}\r\n\r\n{}",
        OK_REPONSE_HEADER,
        content.len(),
        content
    );
    stream.write(response_str.as_bytes()).unwrap();
    stream.flush().unwrap();
}

/**
 * 解析请求结构
 */
fn parse_request(request_str: Cow<str>) {
    let request_lines = request_str.lines();
    for (i, record) in request_lines.enumerate() {
        if !record.contains(":") {
            continue;
        }
        if i == 0 {
            // println!("header: {}", record)
        } else {
            let value = record.replace("\t", "");
            if value.trim().len() == 0 {
                continue;
            }
            let tmp_header_item: Vec<_> = value.split(": ").collect();
            if tmp_header_item[0].len() > 200 {
                println!("too long: {}", tmp_header_item[0])
            }
            // println!("{} - {}", tmp_header_item[0], tmp_header_item[1]);
        }
    }
}

/**
 * 读取文件
 * @param file_path 文件路径
 * @return 文件内容
 */
fn read_html_file(file_path: &str) -> String {
    let content = fs::read_to_string(file_path).unwrap();
    content
}
