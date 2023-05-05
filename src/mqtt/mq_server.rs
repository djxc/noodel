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

/// # 定义服务的接口  
/// ```
/// 1、start启动服务  
/// 2、connect客户端连接事件  
/// 3、request客户端请求事件  
/// 4、close客户端断开连接事件  
/// 5、stop服务器关闭
/// ```
pub trait Server {
    fn start(&self, port: i32) -> bool;
    fn connect(&self);
    fn request(&self, stream: TcpStream);
    fn close(&self);

    fn stop(&self);
}

/// mqttBroker
pub struct MqttBroker {

}

/// 为MqttBroker实现Server特征
impl Server for MqttBroker {

    /// 启动服务  
    /// 
    /// return bool
    /// 
    /// [port] 端口
    /// 
    /// # Example
    /// ```
    /// let mqtt_broker = MqttBroker{};
    /// mqtt_broker.start(8080);
    /// ```
    fn start(&self, port: i32) -> bool {
        println!("mqtt is runing on port:{}", port);
        let result = TcpListener::bind(format!("0.0.0.0:{}", port));
        let listener = match result {
            Ok(listener) => listener,
            Err(error) => {
                panic!("start mqttbroker error on port {}: {}", port, error)
            }
        };
        let listener_cy = listener.try_clone().unwrap();
        thread::spawn(move|| {     
            for stream_result in listener.incoming() {
                let stream = match stream_result {
                    Ok(stream) => stream,
                    Err(error) => {
                        panic!("get stream error:{}", error)
                    }
                };
                // self.request(stream);
                println!("{}", stream.peer_addr().unwrap());
            }     
        });

        loop {
            let client_connect = listener_cy.accept();
            match client_connect {
                Ok((socket, addr)) => {
                    println!("client addr: {} connectd", addr);
                },
                Err(error) => {
                    panic!("connect error:{}", error)
                }
            }
        }

        
        return true;
    }

    fn connect(&self) {
        todo!()
    }

    fn request(&self, mut stream: TcpStream) {
        const OK_REPONSE_HEADER: &str = "HTTP/1.1 200 OK";
        // 定义byte数组，用来存放请求信息；通过stream的read读取stream中的请求到byte数组中
        let mut buffer = [0; 1024];
        stream.read(&mut buffer).unwrap();
        let request: Cow<str> = String::from_utf8_lossy(&buffer[..]);
        println!("{}", stream.peer_addr().unwrap());
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

    fn close(&self) {
        todo!()
    }

    fn stop(&self) {
        todo!()
    }
}

pub fn create_server(thead_num: usize) {
    // 堵塞请求 - 每秒处理请求为上限在90左右，每个请求用时为218ms    
    let port = index_conf::CONF_MAP.get("mqtt_port").unwrap();
    let listener = TcpListener::bind(format!("0.0.0.0:{}", port)).unwrap();
    let ls = Arc::new(listener);
    println!("server is running at :{}", port);
    let ls1 = ls.clone();
    // 需要创一个线程用来接收新客户端的连接   
    // thread::spawn(move || {
        let thread_pool = ThreadPool::new(thead_num);
        for stream in ls1.incoming() {
            let _stream = stream.unwrap();
            thread_pool.execute(|| {
                // hander_connect(_stream);
            });
        };
    // });

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
 * 解析请求结构
 */
fn parse_request(request_str: Cow<str>) {
    let request_lines = request_str.lines();
    for (i, record) in request_lines.enumerate() {
        println!("content: {}", record);
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
