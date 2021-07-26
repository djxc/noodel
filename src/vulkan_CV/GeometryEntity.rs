// 几何类

#[derive(Default, Debug, Clone)]
pub struct Vertex {
    pub position: [f32; 2],
}

// 以给定的点为中心，绘制多个点拼凑成的新点，
pub fn DrawPoint(centerX: f32, centerY: f32, pointSize: i16) -> Vec<Vertex> {
    let mut point = vec![];
    let pointSizeMean = pointSize / 2;
    for i in -pointSizeMean..pointSizeMean {
        for j in -pointSizeMean..pointSizeMean {
            point.push(
                Vertex {
                    position: [centerX + (i as f32 * 0.001), centerY + (j as f32 * 0.001)],
                }
            )
        }
    }
    return point
}