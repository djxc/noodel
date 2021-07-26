pub struct Point {
    pub x:f32,
    pub y:f32,
}

impl Point {
    pub fn show(&self) {
        println!("this point is x: {}; y: {}", self.x, self.y)
    }

    /**
     * **点积**
     */
    pub fn product(&self, point:&Point) -> f32 {
        self.x * point.y - self.y * point.x
    }

    pub fn subtract(&self, point:&Point) -> Point {
        Point{x:self.x-point.x, y: self.y-point.y}
    }

    pub fn toVec(&self) -> Vec<f32>{
        vec![self.x, self.y]
    }

    /**
     * **判断点是否在线段上**
     */
    pub fn inLine(&self, line:Line) -> bool {
        let mut minX = 0.0;
        let mut maxX = 0.0;
        let mut minY = 0.0;
        let mut maxY = 0.0;
        // 获取线段中最大值与最小值
        for point in line.points {
            if point.x > maxX {
                maxX = point.x;
            }
            if point.x < minX {
                minX = point.x;
            }
            if point.y > maxY {
                maxY = point.y
            }
            if point.y < minY {
                minY = point.y
            }
        }
        if self.x < minX || self.x > maxX || self.y < minY || self.y > maxY {
            return false
        }else {
            // let point1_sub = self.subtract(&(line.points[0]));
            // let point1_sub = Point{x: 0.0, y: 1.0};
            // let point2_sub = line.points[1].subtract(&(line.points[0]));
            // if point1_sub.product(&point2_sub) == 0.0 {
            //     return true
            // }else{
            //     return false
            // }
            return false
        }
    }
}

pub struct Line {
    pub points:Vec<Point>,
}

impl Line {
    pub fn show(&self) {
        for point in &self.points {
            point.show()
        }
    }
}
