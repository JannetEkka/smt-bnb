// SMT World — prototype (mock data; real wiring reads smt/ exp records later).
// Visualizes SMT "thinking out loud": 6 personas argue in chat bubbles, the Judge decides.

// ── BSC (BNB Smart Chain) on-chain config ─────────────────────────────────
// After deploying SMTAgentRegistry on BSC (see bnb/ runbook), paste the address +
// agentId here. The badge then links to BscScan and — if ethers loaded — shows the
// LIVE on-chain reputation. Empty contract = "deploy pending" (still valid).
const BSC = {
  network:  "BSC Testnet",
  contract: "",   // ← paste the deployed SMTAgentRegistry (BSC) address after `npm run deploy:testnet`
  agentId:  1,
  rpc:      "https://data-seed-prebsc-1-s1.bnbchain.org:8545",
  explorer: "https://testnet.bscscan.com",
};
const REGISTRY_ABI = ["function reputationBps(uint256) view returns (uint256)"];

function bscBadge(){
  if(BSC.contract){
    return `<a class="onchain ok" target="_blank" rel="noopener"
              href="${BSC.explorer}/address/${BSC.contract}"
              title="Every verdict is written on-chain via SMTAgentRegistry.recordDecision">
              ⛓ Recorded on BSC · <span id="repText">verify ↗</span></a>`;
  }
  return `<span class="onchain pending" title="On-chain agent identity + decision record on BNB Smart Chain">⛓ BNB Chain · on-chain agent</span>`;
}

// Read-only: pull the agent's on-chain reputation (correct/graded, bps) for display.
// No wallet, no keys — just a public RPC read. Never throws into the UI.
async function refreshReputation(){
  if(!BSC.contract || typeof ethers === "undefined") return;
  try{
    const provider = new ethers.JsonRpcProvider(BSC.rpc);
    const c = new ethers.Contract(BSC.contract, REGISTRY_ABI, provider);
    const bps = Number(await c.reputationBps(BSC.agentId));
    const el = document.getElementById("repText");
    // Only surface a % once decisions have actually been graded on-chain (bps>0);
    // otherwise keep the plain "verify ↗" link rather than a misleading 0%.
    if(el && bps > 0) el.textContent = `on-chain accuracy ${(bps/100).toFixed(0)}% · verify ↗`;
  }catch(e){ /* keep the static badge — never break the demo */ }
}

const PERSONA = {
  flow:      {name:"Flow",      pic:"FL", color:"#36c7a0"},
  technical: {name:"Technical", pic:"TC", color:"#7aa2ff"},
  whale:     {name:"Whale",     pic:"WH", color:"#d4af37"},
  onchain:   {name:"OnChain",   pic:"OC", color:"#b98cf0"},
  sentiment: {name:"Sentiment", pic:"SN", color:"#e5685b"},
  regime:    {name:"Regime",    pic:"RG", color:"#9fb4cf"},
};

// Mock per-pair "thinking" — each persona's vote + their in-voice take.
const PAIRS = {
  BTC: {action:"LONG", conf:0.67, why:"Flow + Technical agree LONG; Sentiment vetoed but couldn't drag it under the 0.55 floor; bandit (BTC/LONG/NEUTRAL) nudged 0.67→0.65.",
    votes:{flow:["LONG",.90,"Order book's stacking bids. Buyers in control."],
      technical:["LONG",.80,"Reclaimed the 4H structure. Clean higher-low."],
      whale:["LONG",.62,"Two tracked wallets quietly accumulating off-exchange."],
      onchain:["NEUTRAL",0,"Stablecoin flows flat. Nothing screaming."],
      sentiment:["SHORT",.55,"I'm bearish as always. But I only get a veto. 🙄"],
      regime:["NEUTRAL",0,"Choppy-ish. No strong trend to call."]}},
  ETH: {action:"WAIT", conf:0.41, why:"No two strong personas agree; conviction below the floor. Sitting this one out.",
    votes:{flow:["LONG",.55,"Mild bid lean, nothing convincing."],
      technical:["NEUTRAL",0,"Mid-range. No edge."],
      whale:["NEUTRAL",0,"Quiet wallets today."],
      onchain:["LONG",.40,"Slight Lido inflow uptick."],
      sentiment:["SHORT",.60,"Still doom. Always doom."],
      regime:["NEUTRAL",0,"Ranging."]}},
  BNB: {action:"SHORT", conf:0.71, why:"Flow + Whale align SHORT; a wick-up rejection on the 1H. Bigwick lane.",
    votes:{flow:["SHORT",.85,"Asks heavy. Sellers leaning on it."],
      technical:["SHORT",.70,"Rejected the range high. Wick city."],
      whale:["SHORT",.74,"Big wallet just sent to a CEX deposit address. Distribution."],
      onchain:["NEUTRAL",0,"BSC TVL flat."],
      sentiment:["NEUTRAL",0,"BNB? I stay quiet here."],
      regime:["SHORT",.50,"Rolling over."]}},
  SOL: {action:"LONG", conf:0.63, why:"Technical leads, Flow confirms; fast lane.",
    votes:{flow:["LONG",.72,"Tape's lifting offers."],
      technical:["LONG",.88,"Textbook breakout retest."],
      whale:["NEUTRAL",0,"No big prints."],
      onchain:["LONG",.45,"DEX volume above 30d p80."],
      sentiment:["SHORT",.50,"Overheated, I'd fade it."],
      regime:["LONG",.55,"Trending up."]}},
  DOGE:{action:"WAIT", conf:0.30, why:"Below 200d EMA — LONG is hard-blocked here; nothing strong on the SHORT side. Wait.",
    votes:{flow:["NEUTRAL",0,"Thin. Meme silence."],
      technical:["NEUTRAL",0,"Below the 200d line."],
      whale:["NEUTRAL",0,"Retail-only flow."],
      onchain:["NEUTRAL",0,"Nothing."],
      sentiment:["NEUTRAL",0,"No catalyst, no Musk tweet."],
      regime:["SHORT",.40,"Bleeding."]}},
  XRP:{action:"WAIT",conf:0.38,why:"Mixed. Floor not cleared.",votes:{flow:["LONG",.50,"Slight bid."],technical:["NEUTRAL",0,"Range."],whale:["NEUTRAL",0,"Quiet."],onchain:["NEUTRAL",0,"—"],sentiment:["NEUTRAL",0,"Regulatory pair, I abstain."],regime:["NEUTRAL",0,"Ranging."]}},
  LTC:{action:"WAIT",conf:0.34,why:"Nothing aligned.",votes:{flow:["NEUTRAL",0,"Flat."],technical:["LONG",.45,"Minor coil."],whale:["NEUTRAL",0,"—"],onchain:["NEUTRAL",0,"—"],sentiment:["NEUTRAL",0,"Quiet."],regime:["NEUTRAL",0,"Ranging."]}},
  ADA:{action:"WAIT",conf:0.36,why:"Mixed; below floor.",votes:{flow:["LONG",.48,"Light bid."],technical:["NEUTRAL",0,"Range."],whale:["LONG",.40,"Small accumulation."],onchain:["NEUTRAL",0,"DRep flat."],sentiment:["SHORT",.55,"Contra-pair: euphoria = top."],regime:["NEUTRAL",0,"Ranging."]}},
};

const DRILLS = {
  "Whale flow":"The Whale persona tracks a curated watchlist of top wallets via Etherscan V2 (free). Accumulation off-exchange → bullish; transfers to CEX deposit addresses → distribution → bearish.",
  "Order flow":"Flow reads the WEEX + Binance orderbook composite — our most reliable signal. Bids stacking = buyers in control.",
  "Regime":"Per-pair trend × volatility classification from klines. Drives which lane (fast/bigwick/slow) and the weights.",
  "Why this call":null, // filled per-pair
};

let current = "BTC", tag = "guest";

function enter(mode){
  tag = mode; document.getElementById("userTag").textContent = mode;
  document.getElementById("landing").classList.add("hidden");
  document.getElementById("app").classList.remove("hidden");
  renderPairs(); selectPair("BTC");
}

function renderPairs(){
  const el = document.getElementById("pairList");
  el.innerHTML = Object.entries(PAIRS).map(([p,d])=>
    `<button class="pair-row ${p===current?'active':''}" onclick="selectPair('${p}')">
       <span>${p}</span><span class="dot ${d.action}">${d.action}</span></button>`).join("");
}

function selectPair(p){
  current = p; renderPairs();
  const d = PAIRS[p];
  document.getElementById("pairTitle").textContent = p;
  const v = document.getElementById("verdict");
  v.className = "verdict "+d.action;
  v.innerHTML = `${d.action} <small>conviction ${(d.conf*100).toFixed(0)}%</small>`;
  document.getElementById("onchain").innerHTML = bscBadge();
  refreshReputation();

  // drill chips
  document.getElementById("drill").innerHTML =
    Object.keys(DRILLS).map(k=>`<button class="chip" onclick="drill('${k}')">${k}</button>`).join("");

  // chat bubbles — personas first, Judge last
  const order = ["flow","technical","whale","onchain","sentiment","regime"];
  let html = order.map(k=>{
    const [sig,conf,say] = d.votes[k]; const pr = PERSONA[k];
    const voteCls = sig==="NEUTRAL"?"WAIT":sig;
    return `<div class="bubble">
      <div class="pic" style="background:${pr.color}">${pr.pic}</div>
      <div class="body"><div class="row1"><span class="who">${pr.name}</span>
        <span class="vote dot ${voteCls}">${sig}</span>
        <span class="pct">${conf?(conf*100).toFixed(0)+'%':'—'}</span></div>
        <div class="say">${say}</div></div></div>`;
  }).join("");
  html += `<div class="bubble judge">
      <div class="pic" style="background:linear-gradient(135deg,#f3d98b,#d4af37)">JG</div>
      <div class="body"><div class="row1"><span class="who">The Judge</span>
        <span class="vote dot ${d.action}">${d.action}</span>
        <span class="pct">${(d.conf*100).toFixed(0)}%</span></div>
        <div class="say">${d.why}</div>${BSC.contract
          ? `<div class="onchain-note">⛓ This verdict is recorded on BSC — <a href="${BSC.explorer}/address/${BSC.contract}" target="_blank" rel="noopener">verify ↗</a></div>`
          : ``}</div></div>`;
  document.getElementById("chat").innerHTML = html;
}

function drill(k){
  const body = k==="Why this call" ? PAIRS[current].why : DRILLS[k];
  document.getElementById("popupBody").innerHTML = `<h3>${k} — ${current}</h3><p class="muted" style="line-height:1.5">${body||PAIRS[current].why}</p>`;
  document.getElementById("popup").classList.remove("hidden");
}
function closePopup(){document.getElementById("popup").classList.add("hidden");}

// nav views
document.addEventListener("click",e=>{
  const b = e.target.closest(".nav-item"); if(!b) return;
  document.querySelectorAll(".nav-item").forEach(n=>n.classList.toggle("active",n===b));
  const view = b.dataset.view;
  document.getElementById("view-signals").classList.toggle("hidden",view!=="signals");
  document.getElementById("view-usage").classList.toggle("hidden",view!=="usage");
  document.getElementById("view-copy").classList.toggle("hidden",view!=="copy");
  if(view==="usage") renderUsage();
});

function renderUsage(){
  document.getElementById("usageCards").innerHTML = `
    <div class="card"><h3>WEEX · copy-trade</h3><div class="big">demo</div><p class="muted">Following SMT lead trader. Profit-share & fees shown once live.</p></div>
    <div class="card"><h3>CROO · agent API</h3><div class="big">0</div><p class="muted">Persona-agent calls (pay-per-call USDC). Connect to populate.</p></div>
    <div class="card"><h3>Subscriptions</h3><div class="big">—</div><p class="muted">SMT World premium tier (coming).</p></div>`;
}
