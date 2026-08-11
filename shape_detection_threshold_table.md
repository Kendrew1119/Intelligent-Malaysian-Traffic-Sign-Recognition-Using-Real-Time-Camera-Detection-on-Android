# Shape Detection Thresholds and Decision Rules

| Processing / feature | Threshold or setting | Decision / purpose |
|---|---|---|
| Convex hull | No fixed threshold | Creates a smoothed outer boundary of the selected contour before geometric measurements. |
| Static contour area | > 500 px² | Rejects very small regions/noise in static-image mode. |
| Camera contour area | > 3000 px² | Rejects small noise regions in live-camera mode. |
| Bounding-box aspect ratio | 0.5–1.5 | Keeps approximately square sign-like contours; rejects long narrow regions. |
| Solidity: Red / Blue | > 0.50 | Keeps sufficiently solid contours and rejects irregular background blobs. |
| Solidity: Yellow | > 0.40 | Allows more variation because yellow masks may bleed into the background. |
| Contour smoothing | ε = 0.5% of original perimeter | Smooths serrated contour edges before generating the convex hull. |
| Loose polygon approximation | ε = 3.5% of hull perimeter | Used to identify simple Triangle or Rectangle shapes. |
| Triangle vertices | 3 loose vertices | Classify as Triangle. |
| Four-vertex extent check | Extent < 0.70 | A four-vertex contour is treated as Triangle if it fills less than 70% of its bounding box. |
| Rectangle decision | 4 loose vertices and extent ≥ 0.70 | Classify as Rectangle. |
| Circle-fill ratio | > 0.60 | Treats a contour as sufficiently round for Circle/Octagon classification. |
| Strict polygon approximation | ε = 1% of hull perimeter | Preserves more vertices for the Octagon/Circle decision. |
| Octagon vertices | 7–9 strict vertices | Classify a sufficiently round contour as Octagon. |
| Circle decision | Circle-fill ratio > 0.60 and strict vertices not 7–9 | Classify as Circle. |
| Fallback | All other cases | Classify as Polygon. |
| Camera stability | 3 consecutive frames | Draws the live label only when the same colour and shape remain stable. |
