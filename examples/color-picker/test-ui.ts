import { NeuralNetwork } from "../../src";

export default function setupTestingUI(neuralNetwork: NeuralNetwork) {
	// Add global styles
	const style = document.createElement('style');
	style.textContent = `
		body {
			margin: 0;
			padding: 0;
			font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
			background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
			min-height: 100vh;
			display: flex;
			align-items: center;
			justify-content: center;
		}
	`;
	document.head.appendChild(style);

	// Main container
	const container = document.createElement('div');
	Object.assign(container.style, {
		background: 'white',
		borderRadius: '24px',
		padding: '48px',
		boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)',
		maxWidth: '800px',
		width: '100%',
		margin: '20px'
	});
	document.body.appendChild(container);

	// Header
	const header = document.createElement('div');
	Object.assign(header.style, {
		textAlign: 'center',
		marginBottom: '40px'
	});
	container.appendChild(header);

	const title = document.createElement('h1');
	title.textContent = 'Color Inversion Neural Network';
	Object.assign(title.style, {
		fontSize: '32px',
		fontWeight: '700',
		color: '#1a202c',
		marginBottom: '8px'
	});
	header.appendChild(title);

	const subtitle = document.createElement('p');
	subtitle.textContent = 'Test your neural network\'s color inversion capabilities';
	Object.assign(subtitle.style, {
		color: '#718096',
		fontSize: '16px'
	});
	header.appendChild(subtitle);

	// Input section
	const inputSection = document.createElement('div');
	Object.assign(inputSection.style, {
		display: 'flex',
		flexDirection: 'column',
		alignItems: 'center',
		gap: '16px',
		marginBottom: '40px',
		padding: '24px',
		background: '#f7fafc',
		borderRadius: '16px'
	});
	container.appendChild(inputSection);

	const inputLabel = document.createElement('label');
	inputLabel.textContent = 'INPUT COLOR';
	Object.assign(inputLabel.style, {
		fontSize: '14px',
		fontWeight: '600',
		color: '#4a5568',
		letterSpacing: '0.5px'
	});
	inputSection.appendChild(inputLabel);

	const picker = document.createElement('input');
	picker.type = 'color';
	picker.value = '#808080';
	Object.assign(picker.style, {
		width: '120px',
		height: '120px',
		border: '4px solid white',
		borderRadius: '16px',
		cursor: 'pointer',
		boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
		transition: 'transform 0.2s'
	});
	picker.addEventListener('mouseenter', () => {
		picker.style.transform = 'scale(1.05)';
	});
	picker.addEventListener('mouseleave', () => {
		picker.style.transform = 'scale(1)';
	});
	inputSection.appendChild(picker);

	const hexDisplay = document.createElement('div');
	hexDisplay.textContent = '#808080';
	Object.assign(hexDisplay.style, {
		fontFamily: '"Courier New", monospace',
		fontSize: '18px',
		fontWeight: '600',
		color: '#2d3748',
		marginTop: '12px'
	});
	inputSection.appendChild(hexDisplay);

	// Results section
	const results = document.createElement('div');
	Object.assign(results.style, {
		display: 'grid',
		gridTemplateColumns: '1fr 1fr',
		gap: '24px'
	});
	container.appendChild(results);

	// Ground Truth Card
	const gtCard = document.createElement('div');
	Object.assign(gtCard.style, {
		background: '#f7fafc',
		borderRadius: '16px',
		padding: '24px',
		textAlign: 'center',
		transition: 'transform 0.2s'
	});
	gtCard.addEventListener('mouseenter', () => {
		gtCard.style.transform = 'translateY(-4px)';
	});
	gtCard.addEventListener('mouseleave', () => {
		gtCard.style.transform = 'translateY(0)';
	});
	results.appendChild(gtCard);

	const gtTitle = document.createElement('div');
	gtTitle.textContent = 'GROUND TRUTH';
	Object.assign(gtTitle.style, {
		fontSize: '12px',
		fontWeight: '600',
		color: '#718096',
		letterSpacing: '0.5px',
		marginBottom: '16px'
	});
	gtCard.appendChild(gtTitle);

	const gtSwatch = document.createElement('div');
	Object.assign(gtSwatch.style, {
		width: '100%',
		height: '150px',
		borderRadius: '12px',
		boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
		marginBottom: '16px'
	});
	gtCard.appendChild(gtSwatch);

	const gtHex = document.createElement('div');
	gtHex.textContent = '#7F7F7F';
	Object.assign(gtHex.style, {
		fontFamily: '"Courier New", monospace',
		fontSize: '14px',
		color: '#4a5568',
		fontWeight: '600'
	});
	gtCard.appendChild(gtHex);

	// Network Output Card
	const outCard = document.createElement('div');
	Object.assign(outCard.style, {
		background: '#f7fafc',
		borderRadius: '16px',
		padding: '24px',
		textAlign: 'center',
		transition: 'transform 0.2s'
	});
	outCard.addEventListener('mouseenter', () => {
		outCard.style.transform = 'translateY(-4px)';
	});
	outCard.addEventListener('mouseleave', () => {
		outCard.style.transform = 'translateY(0)';
	});
	results.appendChild(outCard);

	const outTitle = document.createElement('div');
	outTitle.textContent = 'NETWORK OUTPUT';
	Object.assign(outTitle.style, {
		fontSize: '12px',
		fontWeight: '600',
		color: '#718096',
		letterSpacing: '0.5px',
		marginBottom: '16px'
	});
	outCard.appendChild(outTitle);

	const outSwatch = document.createElement('div');
	Object.assign(outSwatch.style, {
		width: '100%',
		height: '150px',
		borderRadius: '12px',
		boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
		marginBottom: '16px'
	});
	outCard.appendChild(outSwatch);

	const outHex = document.createElement('div');
	outHex.textContent = '#7F7F7F';
	Object.assign(outHex.style, {
		fontFamily: '"Courier New", monospace',
		fontSize: '14px',
		color: '#4a5568',
		fontWeight: '600'
	});
	outCard.appendChild(outHex);

	// Update function
	const updateOutputs = async () => {
		const hex = picker.value;
		const r = parseInt(hex.slice(1, 3), 16) / 255;
		const g = parseInt(hex.slice(3, 5), 16) / 255;
		const b = parseInt(hex.slice(5, 7), 16) / 255;
		const input = new Float32Array([r, g, b]);

		const gt = new Float32Array([
			1 - r,
			1 - g,
			1 - b
		]);

		const output = await neuralNetwork.forwardPass(input);

		// Update hex displays
		hexDisplay.textContent = hex.toUpperCase();

		const gtHexValue = `#${Math.round(gt[0] * 255).toString(16).padStart(2, '0')}${Math.round(gt[1] * 255).toString(16).padStart(2, '0')}${Math.round(gt[2] * 255).toString(16).padStart(2, '0')}`.toUpperCase();
		gtHex.textContent = gtHexValue;

		const outHexValue = `#${Math.round(output[0] * 255).toString(16).padStart(2, '0')}${Math.round(output[1] * 255).toString(16).padStart(2, '0')}${Math.round(output[2] * 255).toString(16).padStart(2, '0')}`.toUpperCase();
		outHex.textContent = outHexValue;

		// Update swatches
		gtSwatch.style.backgroundColor = `rgb(${Math.round(gt[0] * 255)}, ${Math.round(gt[1] * 255)}, ${Math.round(gt[2] * 255)})`;
		outSwatch.style.backgroundColor = `rgb(${Math.round(output[0] * 255)}, ${Math.round(output[1] * 255)}, ${Math.round(output[2] * 255)})`;
	};

	picker.addEventListener('input', updateOutputs);

	// Initial update
	updateOutputs();
}