// ────────────────────────────────────────────
//  ST-ECON Live Command Center — App Logic
//  Spatial-Temporal Economic Intelligence UI
// ────────────────────────────────────────────

const API_BASE = window.location.origin;

let apiKey = localStorage.getItem('stecon_api_key');
let orgId  = localStorage.getItem('stecon_org_id');
let projectId = localStorage.getItem('stecon_project_id');

// GNN / Simulation state
let cy = null;
let simulationTimeline = [];   // Array of state snapshots from /simulate_temporal
let currentPrecision = 'FP16';
let selectedNodeId = null;
let latestQueryData = null;

// ─── Default economic knowledge base (Nigerian supply chain example) ───
const DEFAULT_KB_TEXT = `CBN raises interest rates due to rising inflation. 
Lagos Port handles microchip shipments from Taiwan. 
Lagos Port supplies Manufacturing Sector. 
Manufacturing Sector supplies Retail Trade. 
Retail Trade serves MSME Operators. 
CBN regulates Commercial Banks. 
Commercial Banks fund Manufacturing Sector. 
Commodity Markets affect Manufacturing Sector. 
Exchange Rate volatility affects Import Costs. 
Import Costs affect Manufacturing Sector.`;

// ─────────────────────────────────────────────────────────────────
//   UTILITY HELPERS
// ─────────────────────────────────────────────────────────────────

function cleanEntity(text) {
    text = text.replace(/[^a-zA-Z0-9\s]/g, '');
    const stopWords = new Set(['the', 'a', 'an', 'this', 'that', 'these', 'those',
        'its', 'from', 'in', 'of', 'and', 'or', 'is', 'are', 'was', 'were',
        'due', 'to', 'by', 'for', 'on', 'as', 'at', 'with', 'rising']);
    const cleaned = text.split(/\s+/)
        .filter(w => w.trim() !== '' && !stopWords.has(w.toLowerCase()));
    if (cleaned.length === 0) return null;
    return cleaned.slice(0, 4).join('_').toUpperCase().replace(/__+/g, '_');
}

function parseKnowledgeBaseToElements(kbText) {
    const sentences = kbText.split(/[.\n]/);
    const nodes = new Set();
    const edges = [];
    const relationKeywords = [
        'supplies', 'depends on', 'regulates', 'affects', 'impacts',
        'distributes to', 'serves', 'fund', 'handles', 'raises'
    ];

    sentences.forEach(sentence => {
        sentence = sentence.trim();
        if (!sentence || sentence.length < 5) return;
        let matched = false;
        for (const rel of relationKeywords) {
            const relIndex = sentence.toLowerCase().indexOf(rel);
            if (relIndex !== -1) {
                const before = sentence.substring(0, relIndex).trim();
                const after = sentence.substring(relIndex + rel.length).trim();
                const source = cleanEntity(before);
                const target = cleanEntity(after);
                if (source && target && source !== target) {
                    nodes.add(source);
                    nodes.add(target);
                    edges.push({ source, target, rel });
                    matched = true;
                    break;
                }
            }
        }
    });

    const elements = [];
    nodes.forEach(node => elements.push({ data: { id: node, label: node.replace(/_/g, ' '), threat: 0 } }));
    edges.forEach((edge, idx) => elements.push({ data: { id: `e${idx}`, source: edge.source, target: edge.target, rel: edge.rel } }));
    return elements;
}

function getGraphElements() {
    const storedKb = localStorage.getItem('stecon_kb_text') || DEFAULT_KB_TEXT;
    const elements = parseKnowledgeBaseToElements(storedKb);
    if (elements.filter(e => !e.data.source).length > 0) return elements;

    // Fallback hardcoded
    return [
        { data: { id: 'CBN', label: 'CBN', threat: 0 } },
        { data: { id: 'COMMERCIAL_BANKS', label: 'Commercial Banks', threat: 0 } },
        { data: { id: 'LAGOS_PORT', label: 'Lagos Port', threat: 0 } },
        { data: { id: 'MANUFACTURING_SECTOR', label: 'Manufacturing Sector', threat: 0 } },
        { data: { id: 'RETAIL_TRADE', label: 'Retail Trade', threat: 0 } },
        { data: { id: 'MSME_OPERATORS', label: 'MSME Operators', threat: 0 } },
        { data: { id: 'e0', source: 'CBN', target: 'COMMERCIAL_BANKS', rel: 'regulates' } },
        { data: { id: 'e1', source: 'COMMERCIAL_BANKS', target: 'MANUFACTURING_SECTOR', rel: 'funds' } },
        { data: { id: 'e2', source: 'LAGOS_PORT', target: 'MANUFACTURING_SECTOR', rel: 'supplies' } },
        { data: { id: 'e3', source: 'MANUFACTURING_SECTOR', target: 'RETAIL_TRADE', rel: 'supplies' } },
        { data: { id: 'e4', source: 'RETAIL_TRADE', target: 'MSME_OPERATORS', rel: 'serves' } },
    ];
}

function nodeColorFromThreat(threat) {
    if (threat >= 3.0) return '#ff4e6a';   // Danger
    if (threat >= 1.0) return '#ffb740';   // Warning
    return '#0369a1';                       // Normal
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = `toast toast--${type}`;
    toast.innerText = message;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.animation = 'toastOut 0.3s ease forwards';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ─────────────────────────────────────────────────────────────────
//   GRAPH INITIALIZATION
// ─────────────────────────────────────────────────────────────────

function initGraph() {
    const container = document.getElementById('cy');
    if (!container) return;

    const elements = getGraphElements();

    cy = cytoscape({
        container,
        elements,
        style: [
            {
                selector: 'node',
                style: {
                    'background-color': '#0369a1',
                    'label': 'data(label)',
                    'color': '#fff',
                    'text-valign': 'center',
                    'text-outline-width': 1.5,
                    'text-outline-color': '#06070f',
                    'font-family': 'Inter, sans-serif',
                    'font-size': '9px',
                    'font-weight': '600',
                    'width': '55px',
                    'height': '55px',
                    'border-width': '2px',
                    'border-color': 'rgba(255,255,255,0.08)',
                    'transition-property': 'background-color, border-color, border-width, width, height',
                    'transition-duration': '0.35s',
                    'text-wrap': 'wrap',
                    'text-max-width': '60px'
                }
            },
            {
                selector: 'edge',
                style: {
                    'width': 2,
                    'line-color': 'rgba(255,255,255,0.1)',
                    'target-arrow-color': 'rgba(255,255,255,0.1)',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier',
                    'opacity': 0.7,
                    'transition-property': 'line-color, target-arrow-color, opacity',
                    'transition-duration': '0.35s'
                }
            },
            {
                selector: '.selected',
                style: {
                    'border-width': '3.5px',
                    'border-color': '#00d4ff',
                    'width': '70px',
                    'height': '70px',
                    'box-shadow': '0 0 20px rgba(0,212,255,0.4)'
                }
            },
            {
                selector: '.simulating',
                style: {
                    'border-color': '#ffb740',
                    'border-width': '2px',
                }
            }
        ],
        layout: {
            name: 'circle',
            padding: 60,
            animate: true,
            animationDuration: 600
        }
    });

    cy.on('tap', 'node', function(evt) {
        const nodeId = evt.target.id();
        handleNodeSelect(nodeId);
    });
}

function handleNodeSelect(nodeId) {
    if (!cy) return;
    selectedNodeId = nodeId;

    // Update selection highlight
    cy.nodes().removeClass('selected');
    cy.$(`#${nodeId}`).addClass('selected');

    // Fill target input field
    const simTarget = document.getElementById('simulationTarget');
    if (simTarget) simTarget.value = nodeId;

    // Enable simulate button
    const simBtn = document.getElementById('simulateBtn');
    if (simBtn) simBtn.disabled = false;

    showToast(`Node "${nodeId}" selected. Configure shock and simulate.`, 'info');
}

function applyThreatStateToGraph(threatMap) {
    if (!cy || !threatMap) return;
    cy.nodes().forEach(n => {
        const threat = threatMap[n.id()] || 0;
        n.data('threat', threat);
        n.style('background-color', nodeColorFromThreat(threat));
    });
}

function updateGraphFromTimelineStep(stepIndex) {
    if (!simulationTimeline || simulationTimeline.length === 0) return;
    const stepData = simulationTimeline[Math.min(stepIndex, simulationTimeline.length - 1)];
    applyThreatStateToGraph(stepData);

    // Update node count stat
    const nodeCountEl = document.getElementById('statNodes');
    if (nodeCountEl && cy) nodeCountEl.innerText = cy.nodes().length;
}

// ─────────────────────────────────────────────────────────────────
//   HEALTH CHECK
// ─────────────────────────────────────────────────────────────────

async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(4000) });
        if (res.ok) {
            const data = await res.json();
            document.getElementById('statusText').innerText = 'GNN Server Active';
            document.querySelector('.pulse-dot').classList.remove('error');
            document.getElementById('statTenants').innerText = data.tenants || '0';
            document.getElementById('statProjects').innerText = data.total_projects || '0';
        } else throw new Error();
    } catch {
        document.getElementById('statusText').innerText = 'Server Offline';
        document.querySelector('.pulse-dot').classList.add('error');
    }
}

// ─────────────────────────────────────────────────────────────────
//   BACKEND BOOTSTRAP
// ─────────────────────────────────────────────────────────────────

async function bootstrapBackend() {
    try {
        let valid = false;

        if (apiKey) {
            try {
                const res = await fetch(`${API_BASE}/projects`, {
                    headers: { 'x-api-key': apiKey },
                    signal: AbortSignal.timeout(5000)
                });
                if (res.ok) {
                    const data = await res.json();
                    const exists = data.projects.some(p => p.project_id === projectId);
                    if (exists) { valid = true; }
                }
            } catch { /* fall through */ }
        }

        if (!valid) {
            localStorage.clear();

            const resReg = await fetch(`${API_BASE}/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ org_name: 'STECON Demo Org' })
            });
            if (!resReg.ok) throw new Error('Registration failed');
            const dataReg = await resReg.json();
            apiKey = dataReg.api_key;
            orgId  = dataReg.org_id;
            localStorage.setItem('stecon_api_key', apiKey);
            localStorage.setItem('stecon_org_id', orgId);

            const resIngest = await fetch(`${API_BASE}/projects/ingest`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'x-api-key': apiKey },
                body: JSON.stringify({
                    project_name: 'Nigerian Macro Shock Scenario',
                    knowledge_base_text: DEFAULT_KB_TEXT
                })
            });
            if (!resIngest.ok) throw new Error('Ingestion failed');
            const dataIngest = await resIngest.json();
            projectId = dataIngest.project_id;
            localStorage.setItem('stecon_project_id', projectId);
            localStorage.setItem('stecon_kb_text', DEFAULT_KB_TEXT);

            // Reinit graph with default KB
            if (cy) {
                cy.elements().remove();
                cy.add(getGraphElements());
                cy.layout({ name: 'circle', padding: 60 }).run();
            }

            showToast('ST-ECON engine bootstrapped with Nigerian macro scenario', 'success');
        }

        // Update header display
        const keyDisplay = document.getElementById('apiKeyDisplay');
        if (keyDisplay) {
            keyDisplay.innerText = apiKey ? apiKey.substring(0, 22) + '...' : '—';
            keyDisplay.onclick = () => {
                navigator.clipboard.writeText(apiKey);
                showToast('API key copied!', 'info');
            };
        }

        const activeProjectEl = document.getElementById('activeProjectDisplay');
        if (activeProjectEl && projectId) {
            activeProjectEl.innerText = projectId.substring(0, 12) + '...';
        }

        await checkHealth();

    } catch (e) {
        console.error('Bootstrap Error:', e);
        showToast('Error connecting to ST-ECON Engine backend.', 'error');
    }
}

// ─────────────────────────────────────────────────────────────────
//   1. INGESTION
// ─────────────────────────────────────────────────────────────────

async function handleIngestion() {
    const kbText = document.getElementById('kbText').value.trim();
    const name = document.getElementById('projectName').value.trim() || `Project_${Date.now()}`;
    const ingestBtn = document.getElementById('ingestBtn');
    const btnText = ingestBtn.querySelector('.btn-text');

    if (!kbText) { showToast('Please paste knowledge base text first.', 'error'); return; }

    try {
        btnText.textContent = '⏳ Compiling Economic Graph...';
        ingestBtn.disabled = true;

        const res = await fetch(`${API_BASE}/projects/ingest`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'x-api-key': apiKey },
            body: JSON.stringify({ project_name: name, knowledge_base_text: kbText })
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Ingestion API error');
        }

        const data = await res.json();
        projectId = data.project_id;
        localStorage.setItem('stecon_project_id', projectId);
        localStorage.setItem('stecon_kb_text', kbText);

        // Update graph with parsed elements from ingested KB
        const elements = parseKnowledgeBaseToElements(kbText);
        if (cy) {
            cy.elements().remove();
            if (elements.filter(e => !e.data.source).length > 0) {
                cy.add(elements);
            } else {
                cy.add(getGraphElements());
            }
            cy.layout({ name: 'cose', padding: 60, animate: true }).run();
        }

        // Update header
        const activeProjectEl = document.getElementById('activeProjectDisplay');
        if (activeProjectEl) activeProjectEl.innerText = projectId.substring(0, 12) + '...';

        // Reset simulation
        simulationTimeline = [];
        const slider = document.getElementById('timelineSlider');
        if (slider) { slider.value = 0; slider.disabled = true; }
        document.getElementById('timelineStepDisplay').innerText = 'Step: T + 0 months';

        // Update node count
        const nodeCountEl = document.getElementById('statNodes');
        if (nodeCountEl && cy) nodeCountEl.innerText = cy.nodes().length;

        showToast(`✅ GNN compiled! ${data.auto_generated_graph.nodes.length} nodes, ${data.auto_generated_graph.num_edges} edges. Actions extracted: ${data.auto_extracted_actions.join(', ')}.`, 'success');

    } catch (e) {
        showToast(`Ingestion failed: ${e.message}`, 'error');
    } finally {
        btnText.textContent = '🚀 Ingest News & Compile GNN';
        ingestBtn.disabled = false;
    }
}

// ─────────────────────────────────────────────────────────────────
//   2. ODE TEMPORAL SIMULATION (RK4)
// ─────────────────────────────────────────────────────────────────

async function handleSimulate() {
    const targetNode = document.getElementById('simulationTarget').value.trim();
    const shock = parseFloat(document.getElementById('shockMagnitude').value) || 5.0;
    const simBtn = document.getElementById('simulateBtn');
    const slider = document.getElementById('timelineSlider');

    if (!targetNode) { showToast('Select a node from the graph first.', 'error'); return; }
    if (!projectId) { showToast('No active project. Ingest a knowledge base first.', 'error'); return; }

    try {
        simBtn.disabled = true;
        simBtn.querySelector('.btn-text').textContent = '⏳ Running RK4 Solver...';

        // Animate the target node
        cy.nodes().removeClass('simulating');
        cy.$(`#${targetNode}`).addClass('simulating');

        const res = await fetch(`${API_BASE}/projects/${projectId}/simulate_temporal`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'x-api-key': apiKey },
            body: JSON.stringify({ node_name: targetNode, shock_value: shock, steps: 20, dt: 1.0 })
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Simulation failed');
        }

        const data = await res.json();
        simulationTimeline = data.timeline;

        // Enable timeline slider
        slider.disabled = false;
        slider.max = simulationTimeline.length - 1;
        slider.value = 0;

        // Show initial shock state
        updateGraphFromTimelineStep(0);
        document.getElementById('timelineStepDisplay').innerText = 'Step: T + 0 months';

        showToast(`RK4 simulation complete! ${simulationTimeline.length} time steps computed. Drag the timeline to observe stress propagation.`, 'success');

        // Automatically trigger query for full advisory
        await handleNodeQuery(targetNode);

    } catch (e) {
        showToast(`Simulation failed: ${e.message}`, 'error');
        console.error(e);
    } finally {
        simBtn.disabled = false;
        simBtn.querySelector('.btn-text').textContent = '⚡ Solve RK4 Shock Propagation';
        cy.nodes().removeClass('simulating');
    }
}

// ─────────────────────────────────────────────────────────────────
//   3. QUERY: NEURO-SYMBOLIC REASONING (TaCIE + SMT)
// ─────────────────────────────────────────────────────────────────

async function handleNodeQuery(nodeId) {
    if (!projectId) return;

    const smtConsole = document.getElementById('smtConsole');
    const solverTag  = document.getElementById('solverEngineTag');
    const advisoryBody = document.getElementById('advisoryBody');
    const advisoryModelTag = document.getElementById('advisoryModelTag');

    // Prime terminal
    smtConsole.innerHTML = '';
    appendTerminalLine(smtConsole, `> Initializing Neuro-Symbolic query for node: ${nodeId}`, 'system-msg');
    appendTerminalLine(smtConsole, `> Injecting stream signal: shock at ${nodeId}...`, 'system-msg');
    advisoryBody.innerHTML = '<p class="empty-state">⏳ Running Counterfactual Evaluations & Constraint Proofs...</p>';

    try {
        // 1. Stream the signal
        const resStream = await fetch(`${API_BASE}/projects/${projectId}/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'x-api-key': apiKey },
            body: JSON.stringify({ node_name: nodeId, shock_value: 5.0 })
        });
        if (!resStream.ok) throw new Error('Signal streaming failed');

        appendTerminalLine(smtConsole, `  [STREAM OK] Shock propagated through GNN via RK4.`, 'heading-msg');

        // 2. Query full advisory + verify report
        const resQuery = await fetch(`${API_BASE}/projects/${projectId}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'x-api-key': apiKey },
            body: JSON.stringify({ target_node: nodeId, k_hops: 2 })
        });
        if (!resQuery.ok) {
            const err = await resQuery.json();
            throw new Error(err.detail || 'Query failed');
        }

        const data = await resQuery.json();
        latestQueryData = data;

        // ── Render SMT / Symbolic Verifier Terminal ──
        const vr = data.verify_report;
        if (vr) {
            solverTag.textContent = vr.solver_engine || 'Algebraic';
            appendTerminalLine(smtConsole, `> Solver Engine: ${vr.solver_engine}`, 'heading-msg');
            appendTerminalLine(smtConsole, `> Safety Threshold: ${vr.target_threshold}`, 'system-msg');
            appendTerminalLine(smtConsole, `> Resource Budget: ${vr.resource_limit_bytes}`, 'system-msg');
            (vr.proof_log || []).forEach(line => {
                const cls = line.includes('UNSAT') || line.includes('Warning') ? 'warning-msg' :
                            line.includes('Proof') || line.includes('formally') ? 'heading-msg' : '';
                appendTerminalLine(smtConsole, `  ${line}`, cls);
            });
            appendTerminalLine(smtConsole,
                vr.safety_satisfied
                    ? `\n> [SAFETY SATISFIED] ✓ All constraints provably held.`
                    : `\n> [WARNING] Safety bounds NOT provably satisfied.`,
                vr.safety_satisfied ? 'heading-msg' : 'warning-msg'
            );

            // Render commitment ratios
            renderCommitmentRatios(vr.optimized_commitment_ratios || {});
        }

        // ── Render Advisory Body ──
        if (data.advisory) {
            advisoryModelTag.textContent = `${data.advisory_backend || 'Local'} / ${data.advisory_model || 'Rule'}`;
            const paragraphs = data.advisory.split('\n\n');
            advisoryBody.innerHTML = paragraphs.map(p =>
                `<p>${p.replace(/\n/g, '<br>')}</p>`
            ).join('');
        }

        // ── Action confidence bars ──
        renderActionBars(data.counterfactual_actions || {});

        // Update node color from threat map
        if (data.threat_map) {
            applyThreatStateToGraph(data.threat_map);
        }

    } catch (e) {
        appendTerminalLine(smtConsole, `> ERROR: ${e.message}`, 'error-msg');
        advisoryBody.innerHTML = `<p style="color: var(--danger)">Advisory unavailable: ${e.message}</p>`;
        showToast(`Advisory query failed: ${e.message}`, 'error');
    }
}

// ─── Terminal line appender ───
function appendTerminalLine(container, text, className = '') {
    const el = document.createElement('div');
    el.className = `terminal-line ${className}`;
    el.textContent = text;
    container.appendChild(el);
    container.scrollTop = container.scrollHeight;
}

// ─── Render commitment ratios ───
function renderCommitmentRatios(ratios) {
    const container = document.getElementById('optimizedCommitmentRatios');
    container.innerHTML = '';
    const entries = Object.entries(ratios);
    if (entries.length === 0) {
        container.innerHTML = '<span class="empty-state">No actions in scope.</span>';
        return;
    }
    entries.forEach(([action, ratio]) => {
        const chip = document.createElement('div');
        chip.className = 'ratio-chip';
        chip.innerHTML = `${action.replace(/_/g, ' ')} <span>${(ratio * 100).toFixed(0)}%</span>`;
        container.appendChild(chip);
    });
}

// ─── Render action confidence bars ───
function renderActionBars(actions) {
    const section = document.getElementById('actionConfidenceGroup');
    const container = document.getElementById('actionBars');
    container.innerHTML = '';
    const entries = Object.entries(actions);
    if (entries.length === 0) {
        section.style.display = 'none';
        return;
    }
    section.style.display = 'block';
    entries.forEach(([action, confidence]) => {
        const pct = (confidence * 100).toFixed(0);
        const bar = document.createElement('div');
        bar.className = 'action-bar';
        bar.innerHTML = `
            <div class="action-bar__label" title="${action}">${action.replace(/_/g, ' ')}</div>
            <div class="action-bar__track">
                <div class="action-bar__fill" style="width: ${pct}%"></div>
            </div>
            <div class="action-bar__value">${pct}%</div>
        `;
        container.appendChild(bar);
    });
}

// ─────────────────────────────────────────────────────────────────
//   3. TRANSFUSION EDGE TELEMETRY
// ─────────────────────────────────────────────────────────────────

async function handleEdgeInference() {
    const sramLimit = parseInt(document.getElementById('sramLimit').value) || 4096;
    const btn = document.getElementById('edgeInferenceBtn');

    if (!projectId) { showToast('No active project. Ingest a knowledge base first.', 'error'); return; }

    try {
        btn.disabled = true;
        btn.querySelector('.btn-text').textContent = '⏳ Running SRAM Simulation...';

        const [resInference, resAccel] = await Promise.all([
            fetch(`${API_BASE}/projects/${projectId}/edge_inference`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'x-api-key': apiKey },
                body: JSON.stringify({ sram_limit_bytes: sramLimit, precision: currentPrecision })
            }),
            fetch(`${API_BASE}/projects/${projectId}/acceleration`, {
                headers: { 'x-api-key': apiKey }
            })
        ]);

        if (!resInference.ok) throw new Error('Edge inference API error');
        if (!resAccel.ok) throw new Error('Acceleration API error');

        const inferData = await resInference.json();
        const accelData = await resAccel.json();

        updateTelemetryPanel(inferData.edge_inference_simulation, accelData.transfusion_report);
        showToast(`TransFusion simulation complete [${currentPrecision}] — ${inferData.edge_inference_simulation.memory_reduction_ratio}x DRAM savings!`, 'success');

    } catch (e) {
        showToast(`Edge inference failed: ${e.message}`, 'error');
    } finally {
        btn.disabled = false;
        btn.querySelector('.btn-text').textContent = '🎛️ Run Edge Inference Simulation';
    }
}

function updateTelemetryPanel(report, accelReport) {
    // Main stats
    const savingsRatio = report.memory_reduction_ratio;
    document.getElementById('telemetryDramSavings').textContent = `${savingsRatio.toFixed(1)}x`;
    document.getElementById('savingsProgress').style.width = `${Math.min(savingsRatio * 10, 100)}%`;
    document.getElementById('telemetrySramUsage').textContent = report.peak_sram_usage_bytes;
    document.getElementById('telemetryBlockSize').textContent = report.block_size;

    // DPipe / Acceleration
    const dpipe = accelReport.dpipe_scheduling;
    document.getElementById('telemetrySpeedup').textContent = `${dpipe.speedup.toFixed(2)}x`;
    document.getElementById('telemetryLatency').textContent = `${dpipe.pipelined_latency_us.toFixed(2)} μs`;
    document.getElementById('telemetryPe2d').textContent = `${dpipe.pe_2d_util.toFixed(1)}%`;
    document.getElementById('telemetrySoftmaxSavings').textContent =
        `${accelReport.one_pass_attention.memory_reduction.toFixed(1)}x`;
    document.getElementById('telemetryDramSavingsPct').textContent =
        `${accelReport.inter_layer_propagation.dram_savings_pct.toFixed(1)}%`;
}

// ─────────────────────────────────────────────────────────────────
//   PRECISION PILLS INTERACTION
// ─────────────────────────────────────────────────────────────────

function initPrecisionPills() {
    const pills = document.querySelectorAll('.pill-btn');
    pills.forEach(pill => {
        pill.addEventListener('click', () => {
            pills.forEach(p => p.classList.remove('active'));
            pill.classList.add('active');
            currentPrecision = pill.dataset.precision;
        });
    });
}

// ─────────────────────────────────────────────────────────────────
//   TIMELINE SCRUBBER
// ─────────────────────────────────────────────────────────────────

function initTimelineSlider() {
    const slider = document.getElementById('timelineSlider');
    const stepDisplay = document.getElementById('timelineStepDisplay');

    slider.addEventListener('input', () => {
        const step = parseInt(slider.value);
        stepDisplay.innerText = `Step: T + ${step} months`;
        updateGraphFromTimelineStep(step);
    });
}

// ─────────────────────────────────────────────────────────────────
//   MAIN INIT
// ─────────────────────────────────────────────────────────────────

async function init() {
    // Populate default KB text if empty
    const kbTextEl = document.getElementById('kbText');
    if (kbTextEl && !kbTextEl.value.trim()) {
        kbTextEl.value = DEFAULT_KB_TEXT;
    }

    // Init components
    initGraph();
    initPrecisionPills();
    initTimelineSlider();

    // Wire buttons
    document.getElementById('ingestBtn')?.addEventListener('click', handleIngestion);
    document.getElementById('simulateBtn')?.addEventListener('click', handleSimulate);
    document.getElementById('edgeInferenceBtn')?.addEventListener('click', handleEdgeInference);

    // Bootstrap backend
    await bootstrapBackend();

    // Node count
    const nodeCountEl = document.getElementById('statNodes');
    if (nodeCountEl && cy) nodeCountEl.innerText = cy.nodes().length;

    // Periodic health check
    setInterval(checkHealth, 8000);
}

document.addEventListener('DOMContentLoaded', init);
