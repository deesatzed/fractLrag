import React, { useState, useEffect, useRef } from 'react';
import { 
  Terminal, Database, Cpu, ShieldAlert, Zap, Layers, FileText, 
  Image as ImageIcon, Code, Search, ChevronRight, AlertCircle,
  Clock, Activity, HardDrive, MessageSquare, Sliders, GitMerge
} from 'lucide-react';

// --- SOTA CONTRACT DATA ---
const SOTA_CONTRACT = {
  goal: "≥ 98% of local RAG queries resolved in ≤ 400ms (latency) with 0 data exfiltration.",
  constraints: [
    "Hard Limit: Memory overhead must not exceed 85% of unified RAM.",
    "Data Locality: Local LightRAG Postgres Backend + Fractal Latent Derivatives.",
    "Latency: Initial token (TTFT) < 150ms for local MoE routing.",
    "Hardware: Requires Apple M-series (M2 Max+ recommended for 35B models)."
  ],
  failureConditions: [
    "Memory Pressure: Swap usage > 4GB triggers automatic model offloading.",
    "Response Drift: Hallucination rate > 5% on grounded document queries.",
    "API Latency: Retrieval stage > 200ms for collections > 100k chunks."
  ]
};

const RAG_STRATEGIES = [
  { type: "Unified Multimodal", method: "LightRAG 'RAG-Anything' (Native PDF, Tables, Formulas & Images)", icon: <FileText size={16} /> },
  { type: "Fractal Mix-Mode", method: "Macro (Doc) + Meso (Para) + Micro (Sent) Graph-Traversal", icon: <Layers size={16} /> },
  { type: "Latent Derivatives", method: "1st/2nd Order curvature math applied to Vector Chunks", icon: <GitMerge size={16} /> },
  { type: "All-in-One Storage", method: "PostgreSQL (pgvector for embeddings + Apache AGE for Graph)", icon: <HardDrive size={16} /> }
];

// Simple Query Classifier based on fractal_sota_rag_poc.py
const classifyQuery = (q) => {
  const lowerQ = q.toLowerCase();
  if (lowerQ.match(/exact|specific|list|how many|what is the|name the/)) return 'SPECIFICATION';
  if (lowerQ.match(/how does|why|compare|difference|cause|effect|relationship/)) return 'LOGIC';
  if (lowerQ.match(/summarize|overview|briefly|main points|high level/)) return 'SUMMARY';
  return 'SYNTHESIS';
};

const App = () => {
  const [view, setView] = useState('chat');
  const [messages, setMessages] = useState([
    { role: 'system', content: 'Symbio-AI initialized. MLX Core linked to Qwen3.6-35B-A3B.' }
  ]);
  const [input, setInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [healthScore, setHealthScore] = useState(94);
  
  // Profile State based on xai_musk_knowledge_engine_demo.py
  const [docProfile, setDocProfile] = useState({
    accuracy: 'critical',
    complexity: 'high',
    tolerance: 'zero'
  });

  const chatEndRef = useRef(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const simulateGeneration = async (prompt) => {
    setIsGenerating(true);
    const qType = classifyQuery(prompt);
    
    setMessages(prev => [...prev, { 
      role: 'user', 
      content: prompt, 
      metadata: { type: qType, profile: docProfile } 
    }]);
    
    // Simulating MLX Generation + Fractal Retrieval
    setTimeout(() => {
      let response = "";
      let reasoningTrace = {};

      if (qType === 'SPECIFICATION' || docProfile.accuracy === 'critical') {
        reasoningTrace = {
          macro: "Document-level alignment verified (Score: 0.88).",
          meso: "Paragraph context bounded.",
          micro: "Micro-precision (Level 0) hit. Exact terms extracted.",
          deriv: "1st Derivative velocity matched perfectly with exact query parameters."
        };
        response = `CRITICAL ACCURACY MODE ENGAGED. Based on the retrieved multi-source context:\n\nThe exact required parameters have been identified with a 2nd derivative relationship strength of >0.94. No speculation utilized.`;
      } else if (qType === 'LOGIC') {
        reasoningTrace = {
          macro: "Graph-RAG traversed 12 nodes.",
          meso: "Cause/Effect relationship mapped.",
          micro: "Sentence-level extraction.",
          deriv: "High 2nd-order curvature detected (Acceleration of concepts identified)."
        };
        response = `Fractal Logic routing complete. The causal relationship is driven by overlapping latent vectors bridging the paragraph and document structures.`;
      } else {
        reasoningTrace = {
          macro: "Broad document synthesis executed.",
          meso: "Thematic paragraphs merged.",
          micro: "Key sentences weighted.",
          deriv: "Balanced derivative weighting applied."
        };
        response = `Processed via Qwen3.6-35B-A3B. 3B active parameters utilized. System performance stable at 45 tokens/sec. Synthesis complete.`;
      }

      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: response,
        trace: reasoningTrace
      }]);
      setIsGenerating(false);
    }, 1500);
  };

  const Card = ({ title, children, icon: Icon }) => (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 bg-blue-500/10 rounded-lg text-blue-400">
          <Icon size={20} />
        </div>
        <h3 className="text-lg font-semibold text-slate-100">{title}</h3>
      </div>
      {children}
    </div>
  );

  return (
    <div className="min-h-screen bg-slate-950 text-slate-300 font-sans p-6">
      <div className="max-w-7xl mx-auto grid grid-cols-12 gap-6">
        
        {/* SIDEBAR / NAV */}
        <div className="col-span-12 lg:col-span-3 space-y-6">
          <div className="flex items-center gap-3 mb-8">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-cyan-400 rounded-xl flex items-center justify-center text-white shadow-xl shadow-blue-500/20">
              <Zap size={24} strokeWidth={2.5} />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white tracking-tight">Symbio-AI</h1>
              <p className="text-xs text-slate-500 uppercase font-mono">Fractal Latent Edition</p>
            </div>
          </div>

          <nav className="space-y-2">
            {[
              { id: 'chat', label: 'Local Agent', icon: MessageSquare },
              { id: 'dashboard', label: 'System Metrics', icon: Activity },
              { id: 'rag', label: 'Fractal RAG Pipeline', icon: GitMerge },
              { id: 'contract', label: 'Contract Spec', icon: ShieldAlert },
            ].map((item) => (
              <button
                key={item.id}
                onClick={() => setView(item.id)}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
                  view === item.id ? 'bg-blue-600/10 text-blue-400 border border-blue-500/20' : 'hover:bg-slate-900 text-slate-500'
                }`}
              >
                <item.icon size={18} />
                <span className="font-medium">{item.label}</span>
              </button>
            ))}
          </nav>

          {/* SOTA PROFILE CONTROLLER */}
          <div className="pt-6 border-t border-slate-800 space-y-4">
            <div className="flex items-center gap-2 mb-2 text-xs font-mono font-bold text-slate-400 uppercase tracking-wider">
              <Sliders size={14} />
              xAI Document Profile
            </div>
            
            <div className="space-y-3 p-4 bg-slate-900/50 rounded-xl border border-slate-800">
              <div>
                <label className="text-[10px] text-slate-500 uppercase font-bold">Accuracy Need</label>
                <select 
                  value={docProfile.accuracy}
                  onChange={e => setDocProfile({...docProfile, accuracy: e.target.value})}
                  className="w-full mt-1 bg-slate-950 border border-slate-700 rounded p-1.5 text-xs text-blue-400 font-mono outline-none"
                >
                  <option value="critical">Critical (Deriv &gt; 0.9)</option>
                  <option value="high">High</option>
                  <option value="medium">Medium</option>
                </select>
              </div>
              <div>
                <label className="text-[10px] text-slate-500 uppercase font-bold">Hallucination Tolerance</label>
                <select 
                  value={docProfile.tolerance}
                  onChange={e => setDocProfile({...docProfile, tolerance: e.target.value})}
                  className="w-full mt-1 bg-slate-950 border border-slate-700 rounded p-1.5 text-xs text-purple-400 font-mono outline-none"
                >
                  <option value="zero">Zero Tolerance</option>
                  <option value="low">Low</option>
                  <option value="medium">Balanced</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* MAIN CONTENT AREA */}
        <div className="col-span-12 lg:col-span-9 space-y-6">
          
          {view === 'chat' && (
            <div className="flex flex-col h-[85vh] bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden animate-in zoom-in-95 duration-300 shadow-2xl">
              {/* Chat Header */}
              <div className="p-4 border-b border-slate-800 bg-slate-950 flex justify-between items-center">
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                  <span className="text-sm font-mono font-bold text-white">Qwen3.6-35B-A3B (Fractal + LightRAG)</span>
                </div>
                <div className="flex gap-2">
                  <span className="px-2 py-1 bg-slate-800 border border-slate-700 rounded text-[10px] text-slate-400 uppercase font-bold">Local-MLX</span>
                </div>
              </div>

              {/* Chat Body */}
              <div className="flex-1 overflow-y-auto p-6 space-y-6 custom-scrollbar bg-slate-900/50">
                {messages.map((msg, idx) => (
                  <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[85%] rounded-2xl p-5 ${
                      msg.role === 'user' 
                        ? 'bg-blue-600 text-white rounded-tr-none shadow-lg' 
                        : 'bg-slate-800 text-slate-200 rounded-tl-none border border-slate-700 shadow-md'
                    }`}>
                      <div className="flex justify-between items-start mb-2">
                        <div className="flex items-center gap-2">
                          {msg.role === 'assistant' ? <Cpu size={14} className="text-blue-400" /> : <Terminal size={14} />}
                          <span className="text-[10px] uppercase tracking-wider opacity-60 font-bold">
                            {msg.role === 'assistant' ? 'Fractal Agent' : 'Root User'}
                          </span>
                        </div>
                        {msg.metadata && (
                          <div className="px-2 py-0.5 bg-black/20 rounded text-[9px] font-mono font-bold uppercase tracking-widest text-cyan-200 border border-cyan-400/20">
                            Type: {msg.metadata.type}
                          </div>
                        )}
                      </div>
                      
                      <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>

                      {/* Displaying the Fractal Reasoning Trace */}
                      {msg.trace && (
                        <div className="mt-4 pt-4 border-t border-slate-700/50 grid gap-2">
                          <p className="text-[10px] text-slate-500 uppercase font-bold tracking-widest mb-1">Multi-Scale Synthesis Trace</p>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-[11px] font-mono">
                            <div className="bg-slate-900/50 p-2 rounded border border-slate-800">
                              <span className="text-blue-400 font-bold">L2 Macro:</span> {msg.trace.macro}
                            </div>
                            <div className="bg-slate-900/50 p-2 rounded border border-slate-800">
                              <span className="text-green-400 font-bold">L1 Meso:</span> {msg.trace.meso}
                            </div>
                            <div className="bg-slate-900/50 p-2 rounded border border-slate-800">
                              <span className="text-amber-400 font-bold">L0 Micro:</span> {msg.trace.micro}
                            </div>
                            <div className="bg-slate-900/50 p-2 rounded border border-slate-800">
                              <span className="text-purple-400 font-bold">Latent ∂:</span> {msg.trace.deriv}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                
                {isGenerating && (
                  <div className="flex justify-start">
                    <div className="bg-slate-800 p-4 rounded-2xl rounded-tl-none border border-slate-700 animate-pulse flex items-center gap-3">
                      <div className="flex gap-1">
                        <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                        <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                        <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                      </div>
                      <span className="text-xs text-slate-400 font-mono italic">Calculating Latent Derivatives...</span>
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>

              {/* Chat Input */}
              <div className="p-4 bg-slate-950 border-t border-slate-800">
                <form 
                  onSubmit={(e) => {
                    e.preventDefault();
                    if (input.trim() && !isGenerating) {
                      simulateGeneration(input);
                      setInput('');
                    }
                  }}
                  className="relative max-w-4xl mx-auto"
                >
                  <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Ask a specification, logic, or synthesis question..."
                    className="w-full bg-slate-900 border border-slate-700 rounded-xl px-4 py-4 pr-24 focus:outline-none focus:ring-2 focus:ring-blue-500/50 text-sm shadow-inner transition-all"
                  />
                  <div className="absolute right-2 top-2 flex gap-1">
                    <button 
                      type="button"
                      className="p-2 hover:bg-slate-800 text-slate-500 rounded-lg transition-colors"
                    >
                      <ImageIcon size={18} />
                    </button>
                    <button 
                      disabled={isGenerating || !input.trim()}
                      className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 disabled:text-slate-600 text-white rounded-lg font-bold text-sm transition-all shadow-lg shadow-blue-900/20"
                    >
                      Send
                    </button>
                  </div>
                </form>
              </div>
            </div>
          )}

          {view === 'dashboard' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-in fade-in duration-500">
              <Card title="Compute State" icon={Cpu}>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Unified Memory</span>
                    <span className="text-sm font-mono text-blue-400">42.1 / 64 GB</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">GPU Utilization</span>
                    <span className="text-sm font-mono text-purple-400">88%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Latent Space Dim</span>
                    <span className="text-sm font-mono text-cyan-400">D=64 (PoC mode)</span>
                  </div>
                </div>
              </Card>

              <Card title="Storage (Fractal Backend)" icon={HardDrive}>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Macro (L2 Docs)</span>
                    <span className="text-sm font-mono text-white">124 Indexed</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Meso (L1 Paras)</span>
                    <span className="text-sm font-mono text-white">4,821 Indexed</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Micro (L0 Sents + 2nd ∂)</span>
                    <span className="text-sm font-mono text-green-400">18,402 Vectors</span>
                  </div>
                </div>
              </Card>

              <div className="md:col-span-2">
                <Card title="Fractal RAG Orchestration" icon={Database}>
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mt-2">
                    {RAG_STRATEGIES.map((strat, idx) => (
                      <div key={idx} className="p-4 bg-slate-950 border border-slate-800 rounded-xl hover:border-blue-500/50 transition-all group">
                        <div className="text-blue-400 mb-2 group-hover:scale-110 transition-transform">{strat.icon}</div>
                        <h4 className="text-sm font-bold text-slate-100 mb-1">{strat.type}</h4>
                        <p className="text-[10px] text-slate-500 leading-relaxed">{strat.method}</p>
                      </div>
                    ))}
                  </div>
                </Card>
              </div>
            </div>
          )}

          {/* Additional views (RAG, Contract) are similar to previous iteration but inherit the global styling */}
          {view === 'contract' && (
             <div className="bg-blue-600/10 border border-blue-500/20 rounded-xl p-6 animate-in slide-in-from-bottom-4">
               <h2 className="text-xl font-bold text-white mb-2 flex items-center gap-2">
                 <ShieldAlert className="text-blue-400" />
                 The SOTA Architect's Contract
               </h2>
               <p className="text-sm text-slate-400 mb-6 italic">Strict Machine-Parsable Specification for Symbio-AI Organisms.</p>
               <pre className="text-[11px] bg-slate-950 p-4 rounded-lg border border-slate-800 text-cyan-400 font-mono overflow-x-auto">
                 {JSON.stringify(SOTA_CONTRACT, null, 2)}
               </pre>
             </div>
          )}

          {view === 'rag' && (
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-8 text-center flex flex-col items-center justify-center min-h-[400px] animate-in fade-in">
                <Layers size={48} className="text-slate-700 mb-4" />
                <h3 className="text-xl font-bold text-white mb-2">Fractal Ingestion Habitat</h3>
                <p className="text-slate-500 max-w-md mx-auto text-sm">
                  Applying <code>xai_preprocess</code>. Documents are split into L2 (Macro), L1 (Meso), and L0 (Micro). Latent derivatives (velocity & curvature) will be calculated.
                </p>
                <div className="mt-8 flex gap-4">
                  <button className="px-6 py-2 bg-blue-600 text-white rounded-lg text-sm font-bold shadow-lg shadow-blue-900/20 hover:bg-blue-500 transition">Upload High-Stakes Doc</button>
                </div>
            </div>
          )}

        </div>
      </div>

      <style dangerouslySetInnerHTML={{ __html: `
        .custom-scrollbar::-webkit-scrollbar { width: 6px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.5); }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(51, 65, 85, 0.5); border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: rgba(59, 130, 246, 0.3); }
      `}} />
    </div>
  );
};

export default App;
