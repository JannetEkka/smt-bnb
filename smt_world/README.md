# SMT World — user dashboard (prototype)

The user-facing product surface (distinct from the operator's Streamlit builder dashboard). Static
HTML/CSS/JS prototype — **open `index.html` in a browser** (no build step). Mock data for now; the
live version reads SMT's exp records + signals.

## What it shows (per operator direction, 2026-06-15)
- **Landing:** continue as guest · log in · register. **Own SMT login** (exchange-agnostic) — connect
  an exchange *later* to copy-trade + Track Usage.
- **Left sidebar, left-aligned drill-down:** logo top · nav (Signals / Track usage / Copy-trade) ·
  the 8 pairs · **username + avatar pinned bottom-left**. Pick a pair → its panel opens from the left.
- **SMT thinking, in chat bubbles (not charts):** the 6 personas "talk" their take + confidence, then
  **the Judge** bubble delivers the verdict + the ≤500-char "why". Drill chips open popups
  (Whale flow, Order flow, Regime, Why this call).
- **Track usage:** where SMT works for you across platforms (WEEX copy-trade, CROO agent API, subs).

## Next (when wired)
- Feed from real `exp_*.jsonl` + live signals · connect-exchange (WEEX API) for copy-trade + usage ·
  host (GitHub Pages / Vercel) · brand polish + the bubble/cloud iconography from the inspo.
