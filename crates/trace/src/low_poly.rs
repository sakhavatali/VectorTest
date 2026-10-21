//! Low-poly triangulation for raster images.
//!
//! This module builds a jittered sampling grid, adds points along strong image edges and optional
//! foreground boundaries, then connects the points with a Bowyer-Watson Delaunay triangulation.
//! Each triangle is coloured from samples of the source image. This is original Rust code and has
//! no dependency on third-party tracing implementations.

use std::collections::{BTreeMap, HashSet};

use crate::{Raster, TraceError};

const MAX_ANALYSIS_PIXELS: u64 = 1_500_000;
const MAX_MESH_POINTS: usize = 3_500;
const BASE_POINT_LIMIT: usize = 2_300;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LowPolyMode {
    /// Continuous-tone photos and natural images.
    #[default]
    Natural,
    /// Logos and subject silhouettes; infer a foreground mask when possible.
    Silhouette,
    /// Existing vector-like artwork; use a looser, more irregular mesh.
    Illustration,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LowPolyColorMode {
    /// Sample source-image colours.
    #[default]
    Original,
    /// Snap colours to a compact, vivid palette.
    Palette,
    /// Increase colour saturation.
    Vibrant,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LowPolyBoundary {
    /// Use the full image, preserving transparent pixels when present.
    #[default]
    Preserve,
    /// Generate facets over the rectangular image bounds.
    Whole,
    /// Use source alpha or estimate the foreground from border contrast.
    Transparent,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LowPolyParams {
    pub mode: LowPolyMode,
    /// 0 = larger facets; 100 = finer facets.
    pub density: u8,
    /// 0 = fewer edge points; 100 = stronger preservation of visible edges.
    pub edge_preservation: u8,
    pub color_mode: LowPolyColorMode,
    pub boundary: LowPolyBoundary,
    /// Foreground alpha / border-contrast threshold (1..=255).
    pub mask_threshold: u8,
    /// Stable seed so the same parameters produce repeatable geometry.
    pub seed: u64,
}

impl Default for LowPolyParams {
    fn default() -> Self {
        Self {
            mode: LowPolyMode::Natural,
            density: 45,
            edge_preservation: 70,
            color_mode: LowPolyColorMode::Original,
            boundary: LowPolyBoundary::Preserve,
            mask_threshold: 48,
            seed: 1,
        }
    }
}

/// One editable coloured triangular facet in source-raster pixel coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct LowPolyTriangle {
    pub points: [[f64; 2]; 3],
    pub color: [u8; 4],
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Point2 {
    x: f64,
    y: f64,
}

impl Point2 {
    fn midpoint(self, other: Self) -> Self {
        Self { x: (self.x + other.x) * 0.5, y: (self.y + other.y) * 0.5 }
    }

    fn centroid(self, b: Self, c: Self) -> Self {
        Self { x: (self.x + b.x + c.x) / 3.0, y: (self.y + b.y + c.y) / 3.0 }
    }
}

#[derive(Clone, Copy, Debug)]
struct Triangle {
    a: usize,
    b: usize,
    c: usize,
}

/// Turn a raster into a bounded number of editable low-poly facets.
pub fn low_poly(img: &Raster, params: &LowPolyParams) -> Result<Vec<LowPolyTriangle>, TraceError> {
    if img.width < 2 || img.height < 2 {
        return Err(TraceError::Empty);
    }

    let (width, height) = analysis_size(img.width, img.height);
    let scale_x = img.width as f64 / width as f64;
    let scale_y = img.height as f64 / height as f64;
    let image = Raster::from_fn(width, height, |x, y| {
        let sx = (((x as f64 + 0.5) * scale_x).floor() as u32).min(img.width.saturating_sub(1));
        let sy = (((y as f64 + 0.5) * scale_y).floor() as u32).min(img.height.saturating_sub(1));
        img.pixel(sx, sy)
    });

    let mask = make_mask(&image, params);
    let min_side = width.min(height) as f64;
    let density = f64::from(params.density.min(100)) / 100.0;
    let spacing = (min_side * (0.16 * (1.0 - density).powi(2) + 0.025)).max(2.0);
    let row_spacing = (spacing * 0.866_025_403_784_438_6).max(1.0);
    let jitter = spacing
        * match params.mode {
            LowPolyMode::Natural => 0.20,
            LowPolyMode::Silhouette => 0.10,
            LowPolyMode::Illustration => 0.30,
        };

    let mut points = Vec::with_capacity(MAX_MESH_POINTS);
    let mut seen = HashSet::with_capacity(MAX_MESH_POINTS);

    // A staggered and lightly jittered grid avoids the obvious square-cell look of a mosaic.
    let mut row = 0u64;
    let mut y = 0.0;
    while y < height as f64 && points.len() < BASE_POINT_LIMIT {
        let x_offset = if row % 2 == 0 { 0.0 } else { spacing * 0.5 };
        let mut col = 0u64;
        let mut x = x_offset;
        while x < width as f64 && points.len() < BASE_POINT_LIMIT {
            let jx = if x <= 0.0 || x >= (width - 1) as f64 { 0.0 } else { noise(params.seed, col, row) * jitter };
            let jy = if y <= 0.0 || y >= (height - 1) as f64 {
                0.0
            } else {
                noise(params.seed ^ 0xA076_1D64_78BD_642F, row, col) * jitter
            };
            add_point(
                &mut points,
                &mut seen,
                mask.as_deref(),
                width,
                height,
                x + jx,
                y + jy,
                spacing,
                BASE_POINT_LIMIT,
            );
            x += spacing;
            col += 1;
        }
        y += row_spacing;
        row += 1;
    }

    // Pin the corners so natural images cover the full raster.
    for (x, y) in [
        (0.0, 0.0),
        ((width - 1) as f64, 0.0),
        (0.0, (height - 1) as f64),
        ((width - 1) as f64, (height - 1) as f64),
    ] {
        add_point(&mut points, &mut seen, mask.as_deref(), width, height, x, y, spacing, MAX_MESH_POINTS);
    }

    if let Some(mask) = mask.as_deref() {
        add_boundary_points(&mut points, &mut seen, mask, width, height, spacing);
    }
    add_edge_points(
        &mut points,
        &mut seen,
        &image,
        mask.as_deref(),
        spacing,
        params.edge_preservation.min(100),
        params.seed,
    );

    let mesh = triangulate(&points, width, height);
    let mut result = Vec::with_capacity(mesh.len());
    for triangle in mesh {
        let (Some(a), Some(b), Some(c)) =
            (points.get(triangle.a).copied(), points.get(triangle.b).copied(), points.get(triangle.c).copied())
        else {
            continue;
        };
        if orientation(a, b, c).abs() < 1e-7 {
            continue;
        }
        if let Some(mask) = mask.as_deref()
            && !triangle_in_mask(a, b, c, mask, width, height)
        {
            continue;
        }
        let color = apply_color_mode(sample_color(&image, a, b, c), params.color_mode);
        if color[3] == 0 {
            continue;
        }
        result.push(LowPolyTriangle {
            points: [[a.x * scale_x, a.y * scale_y], [b.x * scale_x, b.y * scale_y], [c.x * scale_x, c.y * scale_y]],
            color,
        });
    }
    if result.is_empty() {
        return Err(TraceError::NoTriangles);
    }
    Ok(result)
}

fn analysis_size(width: u32, height: u32) -> (u32, u32) {
    let pixels = u64::from(width) * u64::from(height);
    if pixels <= MAX_ANALYSIS_PIXELS {
        return (width, height);
    }
    let scale = (MAX_ANALYSIS_PIXELS as f64 / pixels as f64).sqrt();
    let mut w = ((width as f64 * scale).floor() as u32).clamp(2, width);
    let mut h = ((height as f64 * scale).floor() as u32).clamp(2, height);
    if u64::from(w) * u64::from(h) > MAX_ANALYSIS_PIXELS {
        if w >= h {
            w = (MAX_ANALYSIS_PIXELS / u64::from(h)).max(2).min(u64::from(w)) as u32;
        } else {
            h = (MAX_ANALYSIS_PIXELS / u64::from(w)).max(2).min(u64::from(h)) as u32;
        }
    }
    (w, h)
}

fn make_mask(img: &Raster, params: &LowPolyParams) -> Option<Vec<bool>> {
    let has_alpha = img.rgba.chunks_exact(4).any(|pixel| pixel.get(3).copied().unwrap_or(255) < 250);
    let infer_from_border = params.mode == LowPolyMode::Silhouette || params.boundary == LowPolyBoundary::Transparent;
    if params.boundary == LowPolyBoundary::Whole {
        return None;
    }
    if has_alpha {
        let threshold = params.mask_threshold.max(1);
        return Some(
            (0..img.height)
                .flat_map(|y| (0..img.width).map(move |x| img.pixel(x, y)[3] >= threshold))
                .collect(),
        );
    }
    if !infer_from_border {
        return None;
    }

    // Estimate a flat background colour from the border; keep pixels whose colour differs enough.
    let mut sums = [0u64; 3];
    let mut count = 0u64;
    for x in 0..img.width {
        for y in [0, img.height - 1] {
            let pixel = img.pixel(x, y);
            for (sum, channel) in sums.iter_mut().zip(pixel.iter().take(3)) {
                *sum += u64::from(*channel);
            }
            count += 1;
        }
    }
    for y in 1..img.height.saturating_sub(1) {
        for x in [0, img.width - 1] {
            let pixel = img.pixel(x, y);
            for (sum, channel) in sums.iter_mut().zip(pixel.iter().take(3)) {
                *sum += u64::from(*channel);
            }
            count += 1;
        }
    }
    let n = count.max(1);
    let background = [sums[0] / n, sums[1] / n, sums[2] / n];
    let threshold = u64::from(params.mask_threshold.max(1)).pow(2);
    Some(
        (0..img.height)
            .flat_map(|y| {
                (0..img.width).map(move |x| {
                    let pixel = img.pixel(x, y);
                    let dr = i64::from(pixel[0]) - background[0] as i64;
                    let dg = i64::from(pixel[1]) - background[1] as i64;
                    let db = i64::from(pixel[2]) - background[2] as i64;
                    (dr * dr + dg * dg + db * db) as u64 >= threshold
                })
            })
            .collect(),
    )
}

/// This helper intentionally keeps the raster, mask, bounds and sampling inputs explicit.
#[allow(clippy::too_many_arguments)]
fn add_point(
    points: &mut Vec<Point2>,
    seen: &mut HashSet<(i32, i32)>,
    mask: Option<&[bool]>,
    width: u32,
    height: u32,
    x: f64,
    y: f64,
    spacing: f64,
    limit: usize,
) {
    if points.len() >= limit || !x.is_finite() || !y.is_finite() {
        return;
    }
    let mut point = Point2 { x: x.clamp(0.0, (width - 1) as f64), y: y.clamp(0.0, (height - 1) as f64) };
    if let Some(mask) = mask
        && !mask_at(mask, width, height, point.x, point.y)
    {
        let Some(snapped) = nearest_foreground(mask, width, height, point, spacing) else { return };
        point = snapped;
    }
    let key = (point.x.round() as i32, point.y.round() as i32);
    if seen.insert(key) {
        points.push(point);
    }
}

fn nearest_foreground(mask: &[bool], width: u32, height: u32, point: Point2, spacing: f64) -> Option<Point2> {
    let radius = ((spacing * 0.4).ceil() as i32).clamp(1, 16);
    let cx = point.x.round() as i32;
    let cy = point.y.round() as i32;
    let mut best = None;
    let mut best_distance = f64::INFINITY;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let x = cx + dx;
            let y = cy + dy;
            if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
                continue;
            }
            if mask_at(mask, width, height, x as f64, y as f64) {
                let distance = (dx * dx + dy * dy) as f64;
                if distance < best_distance {
                    best_distance = distance;
                    best = Some(Point2 { x: x as f64, y: y as f64 });
                }
            }
        }
    }
    best
}

fn mask_at(mask: &[bool], width: u32, height: u32, x: f64, y: f64) -> bool {
    if !x.is_finite() || !y.is_finite() || x < 0.0 || y < 0.0 || x >= width as f64 || y >= height as f64 {
        return false;
    }
    let x = x.round() as u32;
    let y = y.round() as u32;
    if x >= width || y >= height {
        return false;
    }
    mask.get(y as usize * width as usize + x as usize).copied().unwrap_or(false)
}

fn add_boundary_points(
    points: &mut Vec<Point2>,
    seen: &mut HashSet<(i32, i32)>,
    mask: &[bool],
    width: u32,
    height: u32,
    spacing: f64,
) {
    let stride = (spacing * 0.55).round().max(1.0) as u32;
    let mut buckets = HashSet::new();
    for y in 0..height {
        if points.len() >= MAX_MESH_POINTS {
            break;
        }
        for x in 0..width {
            if points.len() >= MAX_MESH_POINTS {
                break;
            }
            if !mask_at(mask, width, height, x as f64, y as f64) {
                continue;
            }
            let edge = x == 0
                || y == 0
                || x + 1 >= width
                || y + 1 >= height
                || !mask_at(mask, width, height, x.saturating_sub(1) as f64, y as f64)
                || !mask_at(mask, width, height, (x + 1) as f64, y as f64)
                || !mask_at(mask, width, height, x as f64, y.saturating_sub(1) as f64)
                || !mask_at(mask, width, height, x as f64, (y + 1) as f64);
            if edge && buckets.insert((x / stride, y / stride)) {
                add_point(points, seen, Some(mask), width, height, x as f64, y as f64, spacing, MAX_MESH_POINTS);
            }
        }
    }
}

fn add_edge_points(
    points: &mut Vec<Point2>,
    seen: &mut HashSet<(i32, i32)>,
    img: &Raster,
    mask: Option<&[bool]>,
    spacing: f64,
    edge_preservation: u8,
    seed: u64,
) {
    let step = (spacing * 0.55).round().max(1.0) as u32;
    let threshold = 105.0 - f64::from(edge_preservation) * 0.8;
    let jitter = spacing * 0.12;
    for y in (0..img.height).step_by(step as usize) {
        if points.len() >= MAX_MESH_POINTS {
            break;
        }
        for x in (0..img.width).step_by(step as usize) {
            if points.len() >= MAX_MESH_POINTS {
                break;
            }
            if edge_strength(img, x, y) < threshold {
                continue;
            }
            let jx = noise(seed ^ 0xE703_7ED1_A0B4_28DB, u64::from(x), u64::from(y)) * jitter;
            let jy = noise(seed ^ 0x8EBC_6AF0_9C88_C6E3, u64::from(y), u64::from(x)) * jitter;
            add_point(
                points,
                seen,
                mask,
                img.width,
                img.height,
                x as f64 + jx,
                y as f64 + jy,
                spacing,
                MAX_MESH_POINTS,
            );
        }
    }
}

fn edge_strength(img: &Raster, x: u32, y: u32) -> f64 {
    let left = img.pixel(x.saturating_sub(1), y);
    let right = img.pixel((x + 1).min(img.width.saturating_sub(1)), y);
    let up = img.pixel(x, y.saturating_sub(1));
    let down = img.pixel(x, (y + 1).min(img.height.saturating_sub(1)));
    let difference = |a: [u8; 4], b: [u8; 4]| {
        let dr = f64::from(a[0]) - f64::from(b[0]);
        let dg = f64::from(a[1]) - f64::from(b[1]);
        let db = f64::from(a[2]) - f64::from(b[2]);
        (dr * dr + dg * dg + db * db).sqrt()
    };
    (difference(left, right) + difference(up, down)) * 0.5
}

fn noise(seed: u64, x: u64, y: u64) -> f64 {
    let mut value = seed ^ x.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ y.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    ((value >> 11) as f64 / 9_007_199_254_740_992.0) * 2.0 - 1.0
}

fn triangle_in_mask(a: Point2, b: Point2, c: Point2, mask: &[bool], width: u32, height: u32) -> bool {
    let center = a.centroid(b, c);
    [a, b, c, center, a.midpoint(b), b.midpoint(c), c.midpoint(a)]
        .into_iter()
        .all(|point| mask_at(mask, width, height, point.x, point.y))
}

fn sample_color(img: &Raster, a: Point2, b: Point2, c: Point2) -> [u8; 4] {
    let center = a.centroid(b, c);
    let samples = [center, center.midpoint(a), center.midpoint(b), center.midpoint(c)];
    let mut sums = [0u64; 4];
    let mut alpha = 0u64;
    for point in samples {
        let x = point.x.floor().clamp(0.0, img.width.saturating_sub(1) as f64) as u32;
        let y = point.y.floor().clamp(0.0, img.height.saturating_sub(1) as f64) as u32;
        let pixel = img.pixel(x, y);
        let a = u64::from(pixel[3]);
        for (sum, channel) in sums.iter_mut().zip(pixel.iter().take(3)) {
            *sum += u64::from(*channel) * a;
        }
        alpha += a;
    }
    if alpha == 0 {
        return [0, 0, 0, 0];
    }
    [
        (sums[0] / alpha).min(255) as u8,
        (sums[1] / alpha).min(255) as u8,
        (sums[2] / alpha).min(255) as u8,
        (alpha / samples.len() as u64).min(255) as u8,
    ]
}

fn apply_color_mode(mut color: [u8; 4], mode: LowPolyColorMode) -> [u8; 4] {
    match mode {
        LowPolyColorMode::Original => color,
        LowPolyColorMode::Palette => {
            const PALETTE: [[u8; 3]; 12] = [
                [18, 25, 44], [31, 73, 135], [18, 150, 164], [48, 157, 91],
                [185, 211, 87], [247, 213, 70], [242, 139, 53], [232, 74, 70],
                [191, 43, 103], [107, 61, 150], [231, 219, 199], [249, 247, 239],
            ];
            let mut nearest = PALETTE[0];
            let mut best = u32::MAX;
            for candidate in PALETTE {
                let dr = i32::from(color[0]) - i32::from(candidate[0]);
                let dg = i32::from(color[1]) - i32::from(candidate[1]);
                let db = i32::from(color[2]) - i32::from(candidate[2]);
                let distance = (dr * dr + dg * dg + db * db) as u32;
                if distance < best {
                    best = distance;
                    nearest = candidate;
                }
            }
            color[0] = nearest[0];
            color[1] = nearest[1];
            color[2] = nearest[2];
            color
        }
        LowPolyColorMode::Vibrant => {
            let luminance = 0.299 * f64::from(color[0]) + 0.587 * f64::from(color[1]) + 0.114 * f64::from(color[2]);
            for channel in color.iter_mut().take(3) {
                *channel = (luminance + (f64::from(*channel) - luminance) * 1.28).round().clamp(0.0, 255.0) as u8;
            }
            color
        }
    }
}

fn triangulate(points: &[Point2], width: u32, height: u32) -> Vec<Triangle> {
    if points.len() < 3 {
        return Vec::new();
    }
    let count = points.len();
    let span = (width as f64 + height as f64).max(1.0) * 16.0;
    let mut work = points.to_vec();
    let super_a = work.len();
    work.push(Point2 { x: -span, y: -span });
    let super_b = work.len();
    work.push(Point2 { x: width as f64 + span, y: -span });
    let super_c = work.len();
    work.push(Point2 { x: width as f64 * 0.5, y: height as f64 + span });

    let mut triangles = vec![Triangle { a: super_a, b: super_b, c: super_c }];
    for inserted in 0..count {
        let Some(point) = work.get(inserted).copied() else { continue };
        let mut bad = vec![false; triangles.len()];
        let mut edges: BTreeMap<(usize, usize), (usize, usize, u32)> = BTreeMap::new();
        let mut found = false;
        for (index, triangle) in triangles.iter().copied().enumerate() {
            let (Some(a), Some(b), Some(c)) = (work.get(triangle.a), work.get(triangle.b), work.get(triangle.c)) else {
                continue;
            };
            if circumcircle_contains(*a, *b, *c, point) {
                if let Some(slot) = bad.get_mut(index) {
                    *slot = true;
                }
                found = true;
                count_edge(&mut edges, triangle.a, triangle.b);
                count_edge(&mut edges, triangle.b, triangle.c);
                count_edge(&mut edges, triangle.c, triangle.a);
            }
        }
        if !found {
            continue;
        }
        let old = std::mem::take(&mut triangles);
        triangles.reserve(old.len().saturating_add(edges.len()));
        for (index, triangle) in old.into_iter().enumerate() {
            if !bad.get(index).copied().unwrap_or(false) {
                triangles.push(triangle);
            }
        }
        for (_, (a, b, edge_count)) in edges {
            if edge_count == 1
                && let Some(triangle) = make_triangle(a, b, inserted, &work)
            {
                triangles.push(triangle);
            }
        }
    }

    triangles
        .into_iter()
        .filter(|triangle| {
            if triangle.a >= count || triangle.b >= count || triangle.c >= count {
                return false;
            }
            let (Some(a), Some(b), Some(c)) = (work.get(triangle.a), work.get(triangle.b), work.get(triangle.c)) else {
                return false;
            };
            orientation(*a, *b, *c).abs() > 1e-7
        })
        .collect()
}

fn count_edge(edges: &mut BTreeMap<(usize, usize), (usize, usize, u32)>, a: usize, b: usize) {
    let key = if a < b { (a, b) } else { (b, a) };
    let edge = edges.entry(key).or_insert((a, b, 0));
    edge.2 = edge.2.saturating_add(1);
}

fn make_triangle(a: usize, b: usize, c: usize, points: &[Point2]) -> Option<Triangle> {
    let (pa, pb, pc) = (points.get(a).copied()?, points.get(b).copied()?, points.get(c).copied()?);
    let area = orientation(pa, pb, pc);
    if !area.is_finite() || area.abs() <= 1e-7 {
        return None;
    }
    if area > 0.0 {
        Some(Triangle { a, b, c })
    } else {
        Some(Triangle { a: b, b: a, c })
    }
}

fn orientation(a: Point2, b: Point2, c: Point2) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

fn circumcircle_contains(a: Point2, b: Point2, c: Point2, point: Point2) -> bool {
    let ax = a.x - point.x;
    let ay = a.y - point.y;
    let bx = b.x - point.x;
    let by = b.y - point.y;
    let cx = c.x - point.x;
    let cy = c.y - point.y;
    let det = (ax * ax + ay * ay) * (bx * cy - by * cx)
        - (bx * bx + by * by) * (ax * cy - ay * cx)
        + (cx * cx + cy * cy) * (ax * by - ay * bx);
    let area = orientation(a, b, c);
    if area > 0.0 { det > 1e-8 } else { det < -1e-8 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_image_generates_repeatable_non_degenerate_facets() {
        let image = Raster::from_fn(96, 72, |_, _| [80, 100, 120, 255]);
        let params = LowPolyParams { density: 25, seed: 19, ..LowPolyParams::default() };
        let first = low_poly(&image, &params).unwrap();
        let second = low_poly(&image, &params).unwrap();
        assert!(!first.is_empty());
        assert_eq!(first, second);
        for triangle in first {
            assert_eq!(triangle.color, [80, 100, 120, 255]);
            let [a, b, c] = triangle.points;
            let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            assert!(area.abs() > 1e-6);
            assert!(triangle.points.iter().flatten().all(|value| value.is_finite()));
        }
    }

    #[test]
    fn opaque_silhouette_keeps_facets_inside_foreground() {
        let image = Raster::from_fn(80, 60, |x, y| {
            if (20..60).contains(&x) && (10..50).contains(&y) { [20, 20, 20, 255] } else { [255, 255, 255, 255] }
        });
        let params = LowPolyParams { mode: LowPolyMode::Silhouette, density: 40, ..LowPolyParams::default() };
        let triangles = low_poly(&image, &params).unwrap();
        assert!(!triangles.is_empty());
        for triangle in triangles {
            let [a, b, c] = triangle.points;
            let x = ((a[0] + b[0] + c[0]) / 3.0).round() as u32;
            let y = ((a[1] + b[1] + c[1]) / 3.0).round() as u32;
            assert!((20..60).contains(&x) && (10..50).contains(&y));
        }
    }

    #[test]
    fn transparent_pixels_do_not_create_opaque_facets() {
        let image = Raster::from_fn(64, 64, |x, y| {
            if (12..52).contains(&x) && (12..52).contains(&y) { [30, 180, 90, 255] } else { [0, 0, 0, 0] }
        });
        let triangles = low_poly(&image, &LowPolyParams::default()).unwrap();
        assert!(!triangles.is_empty());
        assert!(triangles.iter().all(|triangle| triangle.color[3] > 0));
    }

    #[test]
    fn one_pixel_image_returns_an_error() {
        let image = Raster::from_fn(1, 1, |_, _| [0, 0, 0, 255]);
        assert!(matches!(low_poly(&image, &LowPolyParams::default()), Err(TraceError::Empty)));
    }
}
