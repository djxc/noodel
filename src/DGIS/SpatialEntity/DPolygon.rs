use super::DPoint::DPoint;


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
