#!/usr/bin/env python3
import cmath
import math
import re
from collections import defaultdict
import nltk
from nltk.corpus import cmudict
import pronouncing
from flask import Flask, jsonify, render_template_string, request

# Download necessary corpora
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("cmudict", quiet=True)

app = Flask(__name__)

# ==============================================================================
# 1. METRICAL & LEVENSHTEIN DISTANCE VECTOR ENGINES
# ==============================================================================
def edit_distance(s1, s2):
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

# ==============================================================================
# 2. LINE-LEVEL CADENCE & PHONETICS ENGINE
# ==============================================================================
class LineCadenceEngine:
    def __init__(self):
        self.word_db = {}
        self._build_db()

    def _clean(self, text):
        return re.sub(r"[^a-z]", "", text.lower())

    def _extract_profile(self, word):
        phones = pronouncing.phones_for_word(word)
        if not phones:
            return None
        tokens = phones[0].split()
        stresses = "".join([c for t in tokens for c in t if c.isdigit()])
        
        stressed_idx = -1
        for i, t in enumerate(tokens):
            if any(c in t for c in ("1", "2")):
                stressed_idx = i
        
        if stressed_idx == -1:
            rhyme_tail = tokens[-1] if tokens else ""
        else:
            rhyme_tail = "_".join(["".join([c for c in t if not c.isdigit()]) for t in tokens[stressed_idx:]])

        return {
            "word": word,
            "stress": stresses,
            "syllables": len(stresses),
            "rhyme_tail": rhyme_tail
        }

    def _build_db(self):
        words = pronouncing.search(".*")
        for w in words:
            clean_w = self._clean(w)
            if clean_w and clean_w not in self.word_db:
                prof = self._extract_profile(clean_w)
                if prof:
                    self.word_db[clean_w] = prof

    def analyze_phrase(self, phrase):
        words = re.findall(r"\b[a-zA-Z]+\b", phrase.lower())
        full_stress = ""
        tokens = []
        for w in words:
            prof = self.word_db.get(w)
            if prof:
                full_stress += prof["stress"]
                tokens.append(prof)
            else:
                full_stress += "1"
                tokens.append({"word": w, "stress": "1", "syllables": 1, "rhyme_tail": ""})
        return {"full_stress": full_stress, "tokens": tokens}

    def generate_line_candidates(self, target_stress, context_tail=None, limit=12):
        target_len = len(target_stress)
        if target_len == 0:
            return []

        matching_lines = []
        
        def build_phrase(rem_stress, current_phrase, current_tail):
            if len(matching_lines) >= limit * 3:
                return
            if not rem_stress:
                distance = edit_distance(current_tail, context_tail) if context_tail else 0
                matching_lines.append({
                    "line": " ".join([w["word"] for w in current_phrase]),
                    "stress": "".join([w["stress"] for w in current_phrase]),
                    "rhyme_tail": current_tail,
                    "levenshtein_dist": distance
                })
                return

            for chunk_len in range(1, min(5, len(rem_stress) + 1)):
                sub_stress = rem_stress[:chunk_len]
                candidates = [prof for prof in self.word_db.values() if prof["stress"] == sub_stress]
                for cand in candidates[:3]:
                    build_phrase(
                        rem_stress[chunk_len:], 
                        current_phrase + [cand], 
                        cand["rhyme_tail"]
                    )

        build_phrase(target_stress, [], "")

        if context_tail:
            matching_lines.sort(key=lambda x: x["levenshtein_dist"])

        return matching_lines[:limit]

line_engine = LineCadenceEngine()

# ==============================================================================
# 3. BALANCED POLYGON ENGINE (ZERO-CENTROID ORIGIN)
# ==============================================================================
def bjorklund(steps: int, pulses: int) -> list[int]:
    if pulses <= 0: return [0] * steps
    if pulses >= steps: return [1] * steps
    pattern = [[1] for _ in range(pulses)]
    remainder = [[0] for _ in range(steps - pulses)]
    while len(remainder) > 1:
        count = min(len(pattern), len(remainder))
        for i in range(count):
            pattern[i].extend(remainder.pop(0))
    pattern.extend(remainder)
    return [bit for group in pattern for bit in group]

def analyze_polygon(pattern: list[int], N: int) -> dict:
    if not pattern or sum(pattern) == 0:
        return {"pattern_str": "0"*N, "is_balanced": True, "label": f"Empty (0/{N})", "type": "Balanced"}
    total_vector = sum(cmath.exp(1j * 2 * math.pi * i / N) for i, active in enumerate(pattern) if active)
    cx, cy = (total_vector / sum(pattern)).real, (total_vector / sum(pattern)).imag
    balanced = math.hypot(cx, cy) < 1e-5
    return {
        "pattern_str": "".join(map(str, pattern)),
        "is_balanced": balanced,
        "label": f"{'Cyclotomic' if balanced else 'Euclidean'} E({sum(pattern)},{N})",
        "type": "Balanced" if balanced else "Unbalanced"
    }

def get_canonical_rotation(pattern: list[int]) -> tuple[int, ...]:
    n = len(pattern)
    rotations = [tuple(pattern[i:] + pattern[:i]) for i in range(n)]
    return min(rotations)

def get_interesting_polygons(N: int) -> list[dict]:
    results, seen = [], set()
    for k in range(1, N):
        euc_pat = bjorklund(N, k)
        canonical = get_canonical_rotation(euc_pat)
        if canonical not in seen:
            seen.add(canonical)
            results.append(analyze_polygon(list(canonical), N))

    total_combos = 1 << N
    for i in range(1, total_combos - 1):
        pat = [(i >> j) & 1 for j in range(N)]
        analysis = analyze_polygon(pat, N)
        if analysis["is_balanced"]:
            canonical = get_canonical_rotation(pat)
            if canonical not in seen:
                seen.add(canonical)
                results.append(analysis)
    return results

# ==============================================================================
# 4. API ROUTES
# ==============================================================================
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/polygons", methods=["GET"])
def api_polygons():
    n = int(request.args.get("n", 8))
    return jsonify({"n": n, "polygons": get_interesting_polygons(n)})

@app.route("/api/analyze_phrase", methods=["POST"])
def api_analyze_phrase():
    data = request.get_json() or {}
    return jsonify(line_engine.analyze_phrase(data.get("phrase", "")))

@app.route("/api/full_line_stepper", methods=["POST"])
def api_full_line_stepper():
    data = request.get_json() or {}
    candidates = line_engine.generate_line_candidates(
        target_stress=data.get("target_stress", ""),
        context_tail=data.get("context_tail", ""),
        limit=12
    )
    return jsonify({"candidates": candidates})

# ==============================================================================
# 5. UI TEMPLATE
# ==============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Prosodic Cadence Engine & Polygon Weaver</title>
    <style>
        body { font-family: 'Segoe UI', monospace, sans-serif; background: #121214; color: #e0e0e0; margin: 0; padding: 20px; }
        h1, h2, h3 { color: #4db6ac; margin-top: 0; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .card { background: #1e1e24; border-radius: 8px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.4); }
        .full-width { grid-column: span 2; }
        input, button, textarea { background: #2a2a32; border: 1px solid #444; color: #fff; padding: 8px 12px; border-radius: 4px; }
        button { background: #00897b; font-weight: bold; cursor: pointer; }
        button:hover { background: #00bfa5; }
        .button-secondary { background: #37474f; }
        .button-secondary:hover { background: #455a64; }

        .poly-item { padding: 8px; background: #2a2a32; margin-bottom: 4px; cursor: pointer; border-radius: 4px; display: flex; justify-content: space-between; }
        .poly-item:hover { background: #383842; }

        .line-candidate-box { 
            background: #18181c; border: 1px solid #333; padding: 12px; margin-bottom: 8px; 
            border-radius: 6px; display: flex; justify-content: space-between; align-items: center; 
            cursor: pointer; transition: border-color 0.2s;
        }
        .line-candidate-box:hover { border-color: #00bfa5; background: #1f2826; }
        .stress-badge { font-family: monospace; color: #00e676; background: #18281e; padding: 2px 6px; border-radius: 4px; font-size: 12px; }
        .dist-badge { font-size: 11px; color: #ffd54f; margin-left: 8px; }
        .stepper-controls { display: flex; gap: 10px; align-items: center; background: #18181c; padding: 10px; border-radius: 6px; margin-bottom: 15px; flex-wrap: wrap; }
    </style>
</head>
<body>

    <h1>Prosodic Cadence Engine & Polygon Weaver</h1>

    <div class="grid">
        <!-- 1. DECONSTRUCT PHRASE -->
        <div class="card">
            <h2>1. Extract Stress from Phrase</h2>
            <textarea id="phrase-input" style="width: 100%; height: 50px;" placeholder="Type target phrase (e.g. 'silent waters run so deep')..."></textarea>
            <button onclick="analyzePhrase()" style="margin-top: 8px;">Extract Master Meter</button>
            <div id="extracted-stress-display" style="margin-top: 8px; font-family: monospace; color: #ffd54f;"></div>
        </div>

        <!-- 2. POLYGON SELECTOR -->
        <div class="card">
            <h2>2. Select Rhythm Polygon</h2>
            <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 10px;">
                <label>N-gon Syllables:</label>
                <input type="number" id="n-input" value="8" min="3" max="16" style="width: 50px;">
                <button onclick="fetchPolygons()">Find Polygons</button>
            </div>
            <div id="polygon-list" style="max-height: 110px; overflow-y: auto;"></div>
            <div style="margin-top: 8px; display: flex; gap: 8px; align-items: center;">
                <label>Repeat:</label>
                <input type="number" id="repeat-input" value="1" min="1" max="4" style="width: 50px;">
                <button class="button-secondary" onclick="applySelectedPolygon()">Overlay Polygon Meter</button>
            </div>
        </div>

        <!-- 3. FULL-LINE STEPPER CANVAS -->
        <div class="card full-width">
            <h2>3. Line-by-Line Cadence Stepper</h2>
            <p style="font-size: 13px; color: #aaa;">
                Step through full-line offerings matched directly to your active meter and context rhymes.
            </p>

            <div class="stepper-controls">
                <label>Active Meter:</label>
                <input type="text" id="target-stress" value="10010010" style="font-family: monospace; width: 150px;">
                <label>Context Rhyme Tail:</label>
                <input type="text" id="context-tail" placeholder="e.g., IY_P" style="width: 100px;">
                <button onclick="fetchLineCandidates()">Auto-Fit Full Lines ➔</button>
                <button class="button-secondary" onclick="autoExpandSpan(-1)">Contract Meter -1</button>
                <button class="button-secondary" onclick="autoExpandSpan(1)">Expand Meter +1</button>
            </div>

            <h3>Matching Full-Line Offerings</h3>
            <div id="full-line-results">
                <p style="color: #666;">Click "Auto-Fit Full Lines" or pick a polygon / phrase above to load matching candidate lines.</p>
            </div>
        </div>

        <!-- 4. STANZA CANVAS -->
        <div class="card full-width">
            <h2>4. Assembled Stanza Canvas</h2>
            <div id="stanza-lines-list" style="display: flex; flex-direction: column; gap: 8px;"></div>
        </div>
    </div>

    <script>
        let selectedPolygonPattern = "10010010";
        let currentStanza = [];

        async function fetchPolygons() {
            const n = document.getElementById('n-input').value;
            const res = await fetch(`/api/polygons?n=${n}`);
            const data = await res.json();
            const list = document.getElementById('polygon-list');
            list.innerHTML = '';
            
            data.polygons.forEach(p => {
                const el = document.createElement('div');
                el.className = 'poly-item';
                el.innerHTML = `<span><strong>[${p.pattern_str}]</strong> ${p.label}</span> <span style="font-size: 11px; color: ${p.is_balanced ? '#00e676' : '#ff5252'};">${p.type}</span>`;
                el.onclick = () => { 
                    selectedPolygonPattern = p.pattern_str; 
                    applySelectedPolygon();
                };
                list.appendChild(el);
            });
            if(data.polygons.length > 0) selectedPolygonPattern = data.polygons[0].pattern_str;
        }

        function applySelectedPolygon() {
            const reps = parseInt(document.getElementById('repeat-input').value);
            const fullMeter = selectedPolygonPattern.repeat(reps);
            document.getElementById('target-stress').value = fullMeter;
            fetchLineCandidates();
        }

        async function analyzePhrase() {
            const phrase = document.getElementById('phrase-input').value;
            const res = await fetch('/api/analyze_phrase', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ phrase: phrase })
            });
            const data = await res.json();
            if(data.full_stress) {
                document.getElementById('extracted-stress-display').innerText = `Extracted Meter: ${data.full_stress}`;
                document.getElementById('target-stress').value = data.full_stress;
                fetchLineCandidates();
            }
        }

        async function fetchLineCandidates() {
            const stress = document.getElementById('target-stress').value;
            const tail = document.getElementById('context-tail').value;

            const res = await fetch('/api/full_line_stepper', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ target_stress: stress, context_tail: tail })
            });

            const data = await res.json();
            const container = document.getElementById('full-line-results');
            container.innerHTML = '';

            if(!data.candidates || data.candidates.length === 0) {
                container.innerHTML = '<p style="color: #888;">No exact whole-line combinations found for this exact vector. Try expanding or contracting meter span.</p>';
                return;
            }

            data.candidates.forEach(item => {
                const el = document.createElement('div');
                el.className = 'line-candidate-box';
                el.innerHTML = `
                    <div>
                        <strong style="font-size: 16px; color: #fff;">${item.line}</strong>
                        <div style="margin-top: 4px;">
                            <span class="stress-badge">Meter: ${item.stress}</span>
                            <span class="dist-badge">Tail: ${item.rhyme_tail} (Dist: ${item.levenshtein_dist})</span>
                        </div>
                    </div>
                    <button onclick="acceptLine('${item.line.replace(/'/g, "\\'")}', '${item.stress}', '${item.rhyme_tail}')">Accept Line ➔</button>
                `;
                container.appendChild(el);
            });
        }

        function acceptLine(line, stress, tail) {
            currentStanza.push({ line: line, stress: stress, tail: tail });
            document.getElementById('context-tail').value = tail; 
            renderStanza();
            fetchLineCandidates(); 
        }

        function renderStanza() {
            const container = document.getElementById('stanza-lines-list');
            container.innerHTML = '';
            currentStanza.forEach((item, idx) => {
                const el = document.createElement('div');
                el.style.cssText = 'background: #18181c; padding: 10px; border-radius: 6px; display: flex; justify-content: space-between; align-items: center;';
                el.innerHTML = `
                    <div>
                        <span style="color: #00bfa5; font-weight: bold; margin-right: 10px;">Line ${idx + 1}:</span>
                        <span style="font-size: 15px;">${item.line}</span>
                    </div>
                    <span class="stress-badge">${item.stress}</span>
                `;
                container.appendChild(el);
            });
        }

        function autoExpandSpan(delta) {
            const input = document.getElementById('target-stress');
            let val = input.value;
            if(delta > 0) {
                val += '0';
            } else if(val.length > 1) {
                val = val.slice(0, -1);
            }
            input.value = val;
            fetchLineCandidates();
        }

        fetchPolygons();
        fetchLineCandidates();
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    print("Running Restored Master Engine on http://0.0.0.0:5009")
    app.run(host="0.0.0.0", port=5009, debug=True)
