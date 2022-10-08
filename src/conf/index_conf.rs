
/**
 * 参数配置
 */
use std::collections::HashMap;
use lazy_static::lazy_static;


lazy_static! {

    pub static ref CONF_MAP: HashMap<&'static str, &'static str> = {
        let mut map = HashMap::new();
        map.insert("mqtt_port", "8096");
        map
    };

}