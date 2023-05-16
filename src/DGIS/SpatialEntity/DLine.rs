use super::DPoint::DPoint;

/// 线几何体
/// 
/// 线是由多个点按照一定顺序组成
pub struct Line {
    pub points:Vec<DPoint>,
}

impl Line {
    pub fn show(&self) {
        for point in &self.points {
            point.show()
        }
    }
}
