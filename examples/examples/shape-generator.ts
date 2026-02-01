export enum ShapeType {
    CIRCLE = 0,
    SQUARE = 1,
}

export interface ShapeData {
    image: Float32Array;
    label: number;
}

export function generateShape(size: number, type: ShapeType): ShapeData {
    const data = new Float32Array(size * size);
    const center = size / 2;
    
    // Randomize size and position
    const baseSize = (size / 5) + Math.random() * (size / 5);
    // Aspect ratio: 0.7 to 1.3 to allow ovals and rectangles
    const aspect = 0.7 + Math.random() * 0.6;
    const rx = baseSize;
    const ry = baseSize * aspect;
    
    const offsetX = (Math.random() - 0.5) * (size / 2.5);
    const offsetY = (Math.random() - 0.5) * (size / 2.5);
    const centerX = center + offsetX;
    const centerY = center + offsetY;

    const angle = Math.random() * Math.PI * 2;
    const cosA = Math.cos(angle);
    const sinA = Math.sin(angle);

    // Style: Fill (30%) or Stroke (70%) - user draws strokes usually
    const isFilled = Math.random() < 0.3;
    const thickness = 1.0 + Math.random() * 3.0; // 1 to 4 px thickness

    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            // Rotated coordinates relative to center
            const dx = x - centerX;
            const dy = y - centerY;
            const tx = dx * cosA + dy * sinA;
            const ty = -dx * sinA + dy * cosA;

            let dist: number;
            if (type === ShapeType.CIRCLE) {
                // Ellipse equation: distance from center in normalized space
                const normX = tx / rx;
                const normY = ty / ry;
                dist = Math.sqrt(normX * normX + normY * normY);
            } else {
                // Rectangle: Max of normalized tx/rx and ty/ry
                dist = Math.max(Math.abs(tx / rx), Math.abs(ty / ry));
            }

            // Anti-aliasing edge width in normalized units
            const edge = 0.5 / baseSize; 
            let value = 0;

            if (isFilled) {
                if (dist < 1.0 - edge) {
                    value = 1.0;
                } else if (dist < 1.0 + edge) {
                    value = 1.0 - (dist - (1.0 - edge)) / (2 * edge);
                }
            } else {
                // Stroke
                const normalizedHalfThickness = (thickness / 2) / baseSize;
                const distFromEdge = Math.abs(dist - 1.0);
                
                if (distFromEdge < normalizedHalfThickness - edge) {
                    value = 1.0;
                } else if (distFromEdge < normalizedHalfThickness + edge) {
                    value = 1.0 - (distFromEdge - (normalizedHalfThickness - edge)) / (2 * edge);
                }
            }
            data[y * size + x] = Math.max(0, Math.min(1, value));
        }
    }

    return {
        image: data,
        label: type
    };
}

export function generateDataset(count: number, size: number): ShapeData[] {
    const dataset: ShapeData[] = [];
    for (let i = 0; i < count; i++) {
        const type = Math.random() > 0.5 ? ShapeType.CIRCLE : ShapeType.SQUARE;
        dataset.push(generateShape(size, type));
    }
    return dataset;
}
