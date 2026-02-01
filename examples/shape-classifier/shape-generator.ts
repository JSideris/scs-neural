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
    
    // Randomize size and position slightly
    const radius = (size / 4) + Math.random() * (size / 4);
    const offsetX = (Math.random() - 0.5) * (size / 5);
    const offsetY = (Math.random() - 0.5) * (size / 5);
    
    const centerX = center + offsetX;
    const centerY = center + offsetY;

    if (type === ShapeType.CIRCLE) {
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const dx = x - centerX;
                const dy = y - centerY;
                const dist = Math.sqrt(dx * dx + dy * dy);
                
                // Anti-aliased circle
                const edge = 1.0;
                if (dist < radius - edge) {
                    data[y * size + x] = 1.0;
                } else if (dist < radius + edge) {
                    data[y * size + x] = 1.0 - (dist - (radius - edge)) / (2 * edge);
                }
            }
        }
    } else {
        const halfSize = radius;
        const xMin = centerX - halfSize;
        const xMax = centerX + halfSize;
        const yMin = centerY - halfSize;
        const yMax = centerY + halfSize;

        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                // Anti-aliased square
                const edge = 1.0;
                const distToX = Math.min(Math.abs(x - xMin), Math.abs(x - xMax));
                const distToY = Math.min(Math.abs(y - yMin), Math.abs(y - yMax));
                
                const insideX = x >= xMin && x <= xMax;
                const insideY = y >= yMin && y <= yMax;
                
                if (insideX && insideY) {
                    data[y * size + x] = 1.0;
                }
                // Simplified AA for square for now
            }
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
