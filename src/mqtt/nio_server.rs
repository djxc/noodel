
use async_std::net::TcpListener;
use async_std::net::TcpStream;
use std::borrow::Cow;
use std::time::Duration;
use futures::stream::StreamExt;
use async_std::prelude::*;
use async_std::task;

pub fn create_server() {
    // 异步非堵塞请求 - 每秒处理请求为上限在7600左右，每个请求用时为66ms
    let listener = TcpListener::bind("localhost:8096").await.unwrap();
    listener.incoming().for_each_concurrent(/* limit */ None, |tcpstream| async move {
        let tcpstream = tcpstream.unwrap();
        handle_connection(tcpstream).await;
    })
    .await;
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

async fn handle_connection(mut stream: TcpStream) {
    const OK_REPONSE_HEADER: &str = "HTTP/1.1 200 OK";
    let mut buffer = [0; 1024];
    stream.read(&mut buffer).await.unwrap();
    let request = String::from_utf8_lossy(&buffer[..]);
    parse_request(request);
    let content = "hello";
    let response_str = format!(
        "{}\r\nContent-Length:{}\r\n\r\n{}",
        OK_REPONSE_HEADER,
        content.len(),
        content
    );
    task::sleep(Duration::from_millis(10)).await;
    stream.write(response_str.as_bytes()).await.unwrap();
    stream.flush().await.unwrap();
}