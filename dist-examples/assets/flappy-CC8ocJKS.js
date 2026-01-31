import{N as E,A as y}from"./neural-network-D2kPsbY_.js";class _{constructor(i){this.populationSize=i,this.setupUI(),this.birds=[],this.pipes=[],this.frameCount=0}canvas;ctx;birds;pipes;frameCount;CANVAS_WIDTH=800;CANVAS_HEIGHT=600;BIRD_SIZE=20;PIPE_WIDTH=60;PIPE_GAP=150;GRAVITY=.5;FLAP_STRENGTH=-8;PIPE_SPEED=3;PIPE_SPACING=250;telemetryDiv;setupUI(){const i=document.createElement("style");i.textContent=`
			body {
				margin: 0;
				padding: 20px;
				font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
				background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
				display: flex;
				flex-direction: column;
				align-items: center;
				min-height: 100vh;
			}
			canvas {
				border: 4px solid #2d3748;
				border-radius: 8px;
				box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
				background: #87CEEB;
			}
			.telemetry {
				background: white;
				border-radius: 8px;
				padding: 20px;
				margin-top: 20px;
				box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
				width: 800px;
				box-sizing: border-box;
			}
			.telemetry h2 {
				margin: 0 0 15px 0;
				color: #2d3748;
				font-size: 24px;
			}
			.stat-grid {
				display: grid;
				grid-template-columns: repeat(3, 1fr);
				gap: 15px;
			}
			.stat {
				background: #f7fafc;
				padding: 15px;
				border-radius: 6px;
				border-left: 4px solid #667eea;
			}
			.stat-label {
				font-size: 12px;
				color: #718096;
				font-weight: 600;
				text-transform: uppercase;
				letter-spacing: 0.5px;
				margin-bottom: 5px;
			}
			.stat-value {
				font-size: 28px;
				color: #2d3748;
				font-weight: 700;
			}
		`,document.head.appendChild(i),this.canvas=document.createElement("canvas"),this.canvas.width=this.CANVAS_WIDTH,this.canvas.height=this.CANVAS_HEIGHT,this.ctx=this.canvas.getContext("2d"),document.body.appendChild(this.canvas),this.telemetryDiv=document.createElement("div"),this.telemetryDiv.className="telemetry",document.body.appendChild(this.telemetryDiv)}reset(){this.birds=[];for(let i=0;i<this.populationSize;i++)this.birds.push({y:this.CANVAS_HEIGHT/2,velocity:0,isDead:!1,score:0});this.pipes=[];for(let i=0;i<4;i++)this.pipes.push({x:this.CANVAS_WIDTH+i*this.PIPE_SPACING,gapY:Math.random()*(this.CANVAS_HEIGHT-this.PIPE_GAP-100)+50,passed:!1});this.frameCount=0}flap(i){this.birds[i].isDead||(this.birds[i].velocity=this.FLAP_STRENGTH)}update(){this.frameCount++;for(const s of this.pipes)s.x-=this.PIPE_SPEED;this.pipes[this.pipes.length-1].x<this.CANVAS_WIDTH-this.PIPE_SPACING&&this.pipes.push({x:this.CANVAS_WIDTH,gapY:Math.random()*(this.CANVAS_HEIGHT-this.PIPE_GAP-100)+50,passed:!1}),this.pipes=this.pipes.filter(s=>s.x>-this.PIPE_WIDTH);for(let s=0;s<this.birds.length;s++){const e=this.birds[s];if(!e.isDead){if(e.velocity+=this.GRAVITY,e.y+=e.velocity,e.y<0||e.y>this.CANVAS_HEIGHT-this.BIRD_SIZE){e.isDead=!0;continue}for(const t of this.pipes)if(t.x<this.CANVAS_WIDTH/4+this.BIRD_SIZE&&t.x+this.PIPE_WIDTH>this.CANVAS_WIDTH/4){if(e.y<t.gapY||e.y+this.BIRD_SIZE>t.gapY+this.PIPE_GAP){e.isDead=!0;break}t.passed||(e.score++,t.passed=!0)}}}}getBirdState(i){const s=this.birds[i];let e=this.pipes.find(t=>t.x+this.PIPE_WIDTH>this.CANVAS_WIDTH/4);return e||(e=this.pipes[0]),{birdY:s.y,birdVelocity:s.velocity,nextPipeX:e.x-this.CANVAS_WIDTH/4,pipeTopY:e.gapY,pipeBottomY:e.gapY+this.PIPE_GAP}}isDead(i){return this.birds[i].isDead}getScore(i){return this.birds[i].score*100+this.frameCount}render(i,s,e){const t=this.ctx,p=t.createLinearGradient(0,0,0,this.CANVAS_HEIGHT);p.addColorStop(0,"#87CEEB"),p.addColorStop(1,"#E0F6FF"),t.fillStyle=p,t.fillRect(0,0,this.CANVAS_WIDTH,this.CANVAS_HEIGHT),t.fillStyle="#2ECC40",t.strokeStyle="#01FF70",t.lineWidth=3;for(const r of this.pipes)t.fillRect(r.x,0,this.PIPE_WIDTH,r.gapY),t.strokeRect(r.x,0,this.PIPE_WIDTH,r.gapY),t.fillRect(r.x,r.gapY+this.PIPE_GAP,this.PIPE_WIDTH,this.CANVAS_HEIGHT-r.gapY-this.PIPE_GAP),t.strokeRect(r.x,r.gapY+this.PIPE_GAP,this.PIPE_WIDTH,this.CANVAS_HEIGHT-r.gapY-this.PIPE_GAP);let l=0,h=0;for(let r=0;r<this.birds.length;r++){const f=this.birds[r];f.isDead||(l++,h=Math.max(h,f.score),t.fillStyle="rgba(255, 220, 0, 0.3)",t.beginPath(),t.arc(this.CANVAS_WIDTH/4,f.y+this.BIRD_SIZE/2,this.BIRD_SIZE/2,0,Math.PI*2),t.fill())}const u=this.birds.reduce((r,f,c)=>!f.isDead&&f.score>=this.birds[r].score?c:r,0);if(!this.birds[u].isDead){const r=this.birds[u];t.fillStyle="#FFD700",t.strokeStyle="#FFA500",t.lineWidth=2,t.beginPath(),t.arc(this.CANVAS_WIDTH/4,r.y+this.BIRD_SIZE/2,this.BIRD_SIZE/2,0,Math.PI*2),t.fill(),t.stroke(),t.fillStyle="black",t.beginPath(),t.arc(this.CANVAS_WIDTH/4+5,r.y+this.BIRD_SIZE/2-3,2,0,Math.PI*2),t.fill()}const d=s.reduce((r,f)=>r+f.fitness,0)/s.length,o=Math.max(...s.map(r=>r.fitness));this.telemetryDiv.innerHTML=`
			<h2>🧬 Genetic Algorithm Training</h2>
			<div class="stat-grid">
				<div class="stat">
					<div class="stat-label">Generation</div>
					<div class="stat-value">${i}</div>
				</div>
				<div class="stat">
					<div class="stat-label">Alive</div>
					<div class="stat-value">${l}/${this.populationSize}</div>
				</div>
				<div class="stat">
					<div class="stat-label">Best Score</div>
					<div class="stat-value">${h}</div>
				</div>
				<div class="stat">
					<div class="stat-label">Best Fitness (Gen)</div>
					<div class="stat-value">${o.toFixed(0)}</div>
				</div>
				<div class="stat">
					<div class="stat-label">Best Fitness (Ever)</div>
					<div class="stat-value">${e.toFixed(0)}</div>
				</div>
				<div class="stat">
					<div class="stat-label">Avg Fitness</div>
					<div class="stat-value">${d.toFixed(0)}</div>
				</div>
			</div>
		`}}async function x(){const e=new E({layerSizes:[5,8,8,1],trainingBatchSize:1,testingBatchSize:1,outputActivationType:y.LINEAR});try{await e.initialize("xavier")}catch(d){console.error("WebGPU Initialization Failed:",d);const o=document.createElement("div");o.style.color="red",o.style.padding="20px",o.style.background="#fff",o.style.border="1px solid red",o.style.margin="20px",o.innerHTML=`
			<h2>WebGPU Initialization Failed</h2>
			<p>${d.message}</p>
			<p>Your browser or system might not support WebGPU, or hardware acceleration is disabled.</p>
			<p>Try launching Chrome with: <code>--ignore-gpu-blocklist --enable-unsafe-webgpu</code></p>
		`,document.body.prepend(o);return}let t=[];for(let d=0;d<100;d++)t.push(T(e.layerSizes));let p=0,l=0;const h=new _(100);async function u(){p++,h.reset(),t.forEach(c=>{c.fitness=0,c.isAlive=!0});let d=100,o=0;const r=1e4;for(;d>0&&o<r;){o++;const c=[],g=[];for(let n=0;n<100;n++)if(t[n].isAlive){c.push(n);const I=h.getBirdState(n);g.push(new Float32Array([I.birdY/600,(I.birdVelocity+10)/20,I.nextPipeX/400,I.pipeTopY/600,I.pipeBottomY/600]))}if(c.length===0)break;const A=e.layerSizes.length,P=new Array(A),v=new Array(A);P[0]=[],v[0]=[];for(let n=1;n<A;n++)P[n]=c.map(I=>t[I].weights[n]),v[n]=c.map(I=>t[I].biases[n]);const{activations:m}=await e.evaluatePopulation({populationSize:c.length,batchSize:1,weights:P,biases:v,inputs:g,returnActivations:!0});for(let n=0;n<c.length;n++){const I=c[n];m[n]>.5&&h.flap(I)}h.update();for(let n=0;n<100;n++)t[n].isAlive&&h.isDead(n)&&(t[n].isAlive=!1,t[n].fitness=h.getScore(n),d--);h.render(p,t,l),await new Promise(n=>setTimeout(n,1e3/60))}for(let c=0;c<100;c++)t[c].isAlive&&(t[c].fitness=h.getScore(c));const f=t.reduce((c,g)=>g.fitness>c.fitness?g:c);f.fitness>l&&(l=f.fitness,b(f)),console.log(`Generation ${p}: Best=${f.fitness.toFixed(1)}, AllTime=${l.toFixed(1)}`),t=C(t,e.layerSizes),setTimeout(u,100)}u()}function T(a){const i=[null],s=[null];for(let e=1;e<a.length;e++){const t=a[e-1],p=a[e],l=Math.sqrt(2/t),h=new Float32Array(t*p),u=new Float32Array(p);for(let d=0;d<h.length;d++)h[d]=(Math.random()*2-1)*l;for(let d=0;d<u.length;d++)u[d]=(Math.random()*2-1)*.1;i.push(h),s.push(u)}return{weights:i,biases:s,fitness:0,isAlive:!0}}function b(a){return{weights:a.weights.map(i=>i?new Float32Array(i):null),biases:a.biases.map(i=>i?new Float32Array(i):null),fitness:a.fitness,isAlive:a.isAlive}}function C(a,i){const s=[...a].sort((p,l)=>l.fitness-p.fitness),e=Math.floor(a.length*.1),t=s.slice(0,e).map(b);for(;t.length<a.length;){const p=S(s,5),l=S(s,5),h=D(p,l);H(h,.1,.2),t.push(h)}return t}function S(a,i){let s=a[Math.floor(Math.random()*a.length)];for(let e=1;e<i;e++){const t=a[Math.floor(Math.random()*a.length)];t.fitness>s.fitness&&(s=t)}return s}function D(a,i){const s=b(a);for(let e=1;e<s.weights.length;e++){const t=a.weights[e],p=i.weights[e],l=a.biases[e],h=i.biases[e],u=s.weights[e],d=s.biases[e];for(let o=0;o<u.length;o++)u[o]=Math.random()<.5?t[o]:p[o];for(let o=0;o<d.length;o++)d[o]=Math.random()<.5?l[o]:h[o]}return s}function H(a,i,s){for(let e=1;e<a.weights.length;e++){const t=a.weights[e],p=a.biases[e];for(let l=0;l<t.length;l++)Math.random()<i&&(t[l]+=(Math.random()*2-1)*s);for(let l=0;l<p.length;l++)Math.random()<i&&(p[l]+=(Math.random()*2-1)*s)}}x();
