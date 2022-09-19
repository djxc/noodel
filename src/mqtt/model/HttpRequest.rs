/**
 * http请求对象
 */

#[derive(Default, Debug, Clone)]
pub struct HttpRequest {
    pub request_type: str,
    pub url: str,
    pub version: str,
    pub host: str,
}