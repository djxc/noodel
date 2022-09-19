use std::borrow::Cow;
/**
 * 通过tcp启动服务器，实现mqtt协议
 */
use std::fs;
use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
};

pub fn create_server() {
    let listener = TcpListener::bind("localhost:8081").unwrap();
    for stream in listener.incoming() {
        let _stream = stream.unwrap();
        hander_connect(_stream);
        println!("Connection established!")
    }
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
    let content = read_html_file("d:\\index.html");
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
            println!("header: {}", record)
        } else {
            let value = record.replace("\t", "");
            if value.trim().len() == 0 {
                continue;
            }
            let tmp_header_item: Vec<_> = value.split(": ").collect();
            if tmp_header_item[0].len() > 200 {
                println!("too long: {}", tmp_header_item[0])
            }
            println!("{} - {}", tmp_header_item[0], tmp_header_item[1]);
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
