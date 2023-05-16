use super::DPoint::DPoint;

/// 面几何体
/// 
/// 由大于等于3个点按照一定顺序组成的闭合区域
pub struct DPolygon {
    pub points:Vec<DPoint>,
}

impl DPolygon {
    pub fn show(&self) {
        for point in &self.points {
            point.show()
        }
    }
}
