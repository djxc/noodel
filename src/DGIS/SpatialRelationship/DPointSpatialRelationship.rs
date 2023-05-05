/**
 * 点与点、线或面之间的空间关系
 */

struct DPointSpatialRelationship {

}

// /**
//  * **判断点是否在线段上**
//  */
// pub fn inLine(point: DPoint, line:DLine) -> bool {
//     let mut minX = 0.0;
//     let mut maxX = 0.0;
//     let mut minY = 0.0;
//     let mut maxY = 0.0;
//     // 获取线段中最大值与最小值
//     for point in line.points {
//         if point.x > maxX {
//             maxX = point.x;
//         }
//         if point.x < minX {
//             minX = point.x;
//         }
//         if point.y > maxY {
//             maxY = point.y
//         }
//         if point.y < minY {
//             minY = point.y
//         }
//     }
//     if self.x < minX || self.x > maxX || self.y < minY || self.y > maxY {
//         return false
//     }else {
//         // let point1_sub = self.subtract(&(line.points[0]));
//         // let point1_sub = Point{x: 0.0, y: 1.0};
//         // let point2_sub = line.points[1].subtract(&(line.points[0]));
//         // if point1_sub.product(&point2_sub) == 0.0 {
//         //     return true
//         // }else{
//         //     return false
//         // }
//         return false
//     }
// }