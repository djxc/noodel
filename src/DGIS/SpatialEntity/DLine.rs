use super::DPoint::DPoint;

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
