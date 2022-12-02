pub struct DPoint {
    pub x:f32,
    pub y:f32,
}

impl DPoint {
    pub fn show(&self) {
        println!("this point is x: {}; y: {}", self.x, self.y)
    }

    /**
     * **点积**
     */
    pub fn product(&self, point:&DPoint) -> f32 {
        self.x * point.y - self.y * point.x
    }

    pub fn subtract(&self, point:&DPoint) -> DPoint {
        DPoint{x:self.x-point.x, y: self.y-point.y}
    }

    pub fn toVec(&self) -> Vec<f32>{
        vec![self.x, self.y]
    }   
}