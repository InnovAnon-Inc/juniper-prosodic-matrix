#!/usr/bin/env python3
import cmath
import math
import re
import random
from collections import defaultdict
import nltk
from nltk.corpus import cmudict
import pronouncing
from flask import Flask, jsonify, render_template_string, request

# Ensure required corpora are ready
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("cmudict", quiet=True)

app = Flask(__name__)

# ==============================================================================
# 1. GRAMMAR & CADENCE ENGINE
# ==============================================================================
VALID_POS_TRANSITIONS = {
    'DT': {'JJ', 'NN', 'NNS', 'RB'},
    'JJ': {'NN', 'NNS', 'JJ', 'CC'},
    'NN': {'VBZ', 'VBD', 'VBP', 'IN', 'CC', 'RB', '.'},
    'NNS': {'VBP', 'VBD', 'IN', 'CC', 'RB'},
    'RB': {'JJ', 'VB', 'VBD', 'VBG', 'RB'},
    'VB': {'DT', 'JJ', 'NN', 'NNS', 'RB', 'IN'},
    'VBD': {'DT', 'JJ', 'NN', 'NNS', 'RB', 'IN'},
    'VBZ': {'DT', 'JJ', 'NN', 'NNS', 'RB', 'IN'},
    'VBP': {'DT', 'JJ', 'NN', 'NNS', 'RB', 'IN'},
    'IN': {'DT', 'JJ', 'NN', 'NNS'}
}

def edit_distance(s1, s2):
    if len(s1) < len(s2): return edit_distance(s2, s1)
    if len(s2) == 0: return len(s1)
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

class UnifiedCadenceEngine:
    def __init__(self):
        self.word_db = {}
        self._build_db()

    def _clean(self, text):
        return re.sub(r"[^a-z]", "", text.lower())

    def _extract_profile(self, word):
        phones = pronouncing.phones_for_word(word)
        if not phones: return None
        tokens = phones[0].split()
        stresses = "".join([c for t in tokens for c in t if c.isdigit()])
        
        vowels = [t for t in tokens if any(c.isdigit() for c in t)]
        vowel_kernel = vowels[0] if vowels else ""
        
        return {
            "word": word,
            "stress": stresses,
            "syllables": len(stresses),
            "rhyme_tail": "_".join([t for t in tokens if not any(c.isdigit() for c in t)]),
            "vowel_kernel": re.sub(r'\d', '', vowel_kernel)
        }

    def _build_db(self):
        words = pronouncing.search(".*")
        for w in words:
            clean_w = self._clean(w)
            if clean_w and clean_w not in self.word_db:
                prof = self._extract_profile(clean_w)
                if prof: self.word_db[clean_w] = prof

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
                tokens.append({"word": w, "stress": "1", "syllables": 1, "rhyme_tail": "", "vowel_kernel": ""})
        return {"full_stress": full_stress, "tokens": tokens}

    def validate_line_grammar(self, word_list):
        tagged = nltk.pos_tag(word_list)
        score = 0
        for i in range(len(tagged) - 1):
            p1, p2 = tagged[i][1][:2], tagged[i+1][1][:2]
            if p2 in VALID_POS_TRANSITIONS.get(p1, {p2}):
                score += 1
        return score / max(1, len(word_list) - 1)

    def generate_line_candidates(self, target_stress, context_tail=None, page=0, page_size=10):
        target_len = len(target_stress)
        if target_len == 0: return []

        matching_lines = []
        db_items = list(self.word_db.values())
        random.seed(page + 42)
        random.shuffle(db_items)

        def build_phrase(rem_stress, current_phrase):
            if len(matching_lines) >= (page + 1) * page_size * 5:
                return
            if not rem_stress:
                words = [w["word"] for w in current_phrase]
                g_score = self.validate_line_grammar(words)
                vowels = [w["vowel_kernel"] for w in current_phrase if w["vowel_kernel"]]
                internal_rhyme = len(vowels) - len(set(vowels))
                
                last_tail = current_phrase[-1]["rhyme_tail"]
                tail_dist = edit_distance(last_tail, context_tail) if context_tail else 0

                if g_score >= 0.4:
                    matching_lines.append({
                        "line": " ".join(words),
                        "stress": "".join([w["stress"] for w in current_phrase]),
                        "rhyme_tail": last_tail,
                        "levenshtein_dist": tail_dist,
                        "grammar_score": round(g_score, 2),
                        "internal_rhyme_bonus": internal_rhyme
                    })
                return

            for chunk_len in range(1, min(5, len(rem_stress) + 1)):
                sub_stress = rem_stress[:chunk_len]
                candidates = [prof for prof in db_items if prof["stress"] == sub_stress]
                for cand in candidates[:3]:
                    build_phrase(rem_stress[chunk_len:], current_phrase + [cand])

        build_phrase(target_stress, [])
        matching_lines.sort(key=lambda x: (-x["grammar_score"], -x["internal_rhyme_bonus"], x["levenshtein_dist"]))
        
        start_idx = page * page_size
        return matching_lines[start_idx : start_idx + page_size]

engine = UnifiedCadenceEngine()

# ==============================================================================
# 2. POLYGON GENERATOR (FORWARD/REVERSE)
# ==============================================================================
def analyze_polygon_with_reverse(pattern: list[int], N: int) -> dict:
    pat_str = "".join(map(str, pattern))
    rev_str = pat_str[::-1]
    
    total_vector = sum(cmath.exp(1j * 2 * math.pi * i / N) for i, active in enumerate(pattern) if active)
    cx, cy = (total_vector / sum(pattern)).real, (total_vector / sum(pattern)).imag if sum(pattern) > 0 else (0,0)
    balanced = math.hypot(cx, cy) < 1e-5
    
    return {
        "pattern_str": pat_str,
        "reverse_str": rev_str,
        "is_palindrome": pat_str == rev_str,
        "is_balanced": balanced,
        "label": f"E({sum(pattern)},{N})"
    }

def get_filtered_polygons(N: int, filter_type="all") -> list[dict]:
    results = []
    total_combos = 1 << N
    for i in range(1, total_combos):
        pat = [(i >> j) & 1 for j in range(N)]
        analysis = analyze_polygon_with_reverse(pat, N)
        if filter_type == "balanced" and not analysis["is_balanced"]: continue
        if filter_type == "palindromic" and not analysis["is_palindrome"]: continue
        results.append(analysis)
    return results[:30]

# ==============================================================================
# 3. ROUTES
# ==============================================================================
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/polygons", methods=["GET"])
def api_polygons():
    n = int(request.args.get("n", 8))
    f = request.args.get("filter", "all")
    return jsonify({"polygons": get_filtered_polygons(n, f)})

@app.route("/api/analyze_phrase", methods=["POST"])
def api_analyze_phrase():
    data = request.get_json() or {}
    return jsonify(engine.analyze_phrase(data.get("phrase", "")))

@app.route("/api/full_line_stepper", methods=["POST"])
def api_full_line_stepper():
    data = request.get_json() or {}
    candidates = engine.generate_line_candidates(
        target_stress=data.get("target_stress", ""),
        context_tail=data.get("context_tail", ""),
        page=int(data.get("page", 0)),
        page_size=10
    )
    return jsonify({"candidates": candidates})

# ==============================================================================
# 4. UNIFIED UI TEMPLATE
# ==============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Prosodic Studio & Polygon Weaver</title>
    <style>
        body { font-family: 'Segoe UI', monospace; background: #121214; color: #e0e0e0; margin: 0; padding: 20px; }
        h1, h2, h3 { color: #4db6ac; margin-top: 0; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .card { background: #1e1e24; border-radius: 8px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.4); }
        .full-width { grid-column: span 2; }
        input, button, textarea, select { background: #2a2a32; border: 1px solid #444; color: #fff; padding: 8px 12px; border-radius: 4px; }
        button { background: #00897b; font-weight: bold; cursor: pointer; }
        button:hover { background: #00bfa5; }
        .button-secondary { background: #37474f; }

        .matrix-bar { display: flex; gap: 4px; margin: 15px 0; overflow-x: auto; padding: 10px; background: #18181c; border-radius: 6px; }
        .syllable-box { 
            flex: 1; min-width: 40px; height: 50px; display: flex; flex-direction: column; 
            align-items: center; justify-content: center; background: #2a2a32; 
            border: 1px solid #444; border-radius: 4px; cursor: pointer; user-select: none;
        }
        .syllable-box.stress-1 { color: #00e676; font-weight: bold; }
        .syllable-box.stress-2 { color: #ffb74d; font-weight: bold; }
        .syllable-box.stress-0 { color: #666; }

        .poly-item { padding: 8px; background: #2a2a32; margin-bottom: 4px; cursor: pointer; border-radius: 4px; display: flex; justify-content: space-between; }
        .poly-item:hover { background: #383842; }

        .line-candidate-box { 
            background: #18181c; border: 1px solid #333; padding: 12px; margin-bottom: 8px; 
            border-radius: 6px; display: flex; justify-content: space-between; align-items: center; 
        }
        .badge { font-family: monospace; color: #00e676; background: #18281e; padding: 2px 6px; border-radius: 4px; font-size: 11px; margin-right: 6px; }
        .stepper-controls { display: flex; gap: 10px; align-items: center; background: #18181c; padding: 10px; border-radius: 6px; margin-bottom: 15px; flex-wrap: wrap; }
    </style>
</head>
<body>

    <h1>Prosodic Studio & Polygon Weaver</h1>

    <div class="grid">
        <!-- 1. DECONSTRUCT PHRASE -->
        <div class="card">
            <h2>1. Deconstruct Target Phrase</h2>
            <textarea id="phrase-input" style="width: 100%; height: 45px;" placeholder="Type phrase to extract stress..."></textarea>
            <button onclick="analyzePhrase()" style="margin-top: 8px;">Extract Master Meter</button>
            <div id="extracted-stress-display" style="margin-top: 8px; font-family: monospace; color: #ffd54f;"></div>
        </div>

        <!-- 2. POLYGON SELECTOR & FILTERS -->
        <div class="card">
            <h2>2. Polygon Selector & Reverse Filter</h2>
            <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 10px;">
                <label>N-gon:</label>
                <input type="number" id="n-input" value="8" min="3" max="16" style="width: 45px;">
                <select id="poly-filter" onchange="fetchPolygons()">
                    <option value="all">All Shapes</option>
                    <option value="balanced">Zero-Centroid Only</option>
                    <option value="palindromic">Palindromic Only</option>
                </select>
                <button onclick="fetchPolygons()">Find</button>
            </div>
            <div id="polygon-list" style="max-height: 100px; overflow-y: auto;"></div>
        </div>

        <!-- 3. PROSODIC MATRIX -->
        <div class="card full-width">
            <h2>3. Interactive Prosodic Matrix</h2>
            <p style="font-size: 12px; color: #aaa;">Click to toggle stress levels (0 = Unstressed, 1 = Primary, 2 = Secondary).</p>
            <div id="matrix-container" class="matrix-bar"></div>
        </div>

        <!-- 4. FULL-LINE STEPPER WITH PAGINATION -->
        <div class="card full-width">
            <h2>4. Full-Line Cadence Offerings</h2>
            <div class="stepper-controls">
                <label>Active Meter:</label>
                <input type="text" id="target-stress" value="10010010" style="font-family: monospace; width: 140px;">
                <label>Context Rhyme Tail:</label>
                <input type="text" id="context-tail" placeholder="e.g. AH_NG" style="width: 90px;">
                <button onclick="fetchLineCandidates(0)">Search Lines</button>
                <button class="button-secondary" onclick="autoExpandSpan(-1)">Contract -1</button>
                <button class="button-secondary" onclick="autoExpandSpan(1)">Expand +1</button>
            </div>

            <div id="full-line-results"></div>

            <div style="margin-top: 15px; display: flex; gap: 10px; align-items: center;">
                <button class="button-secondary" onclick="changePage(-1)">◄ Previous Batch</button>
                <span id="page-display" style="font-weight: bold; color: #ffd54f;">Batch 1</span>
                <button class="button-secondary" onclick="changePage(1)">Next Batch ➔</button>
            </div>
        </div>

        <!-- 5. STANZA CANVAS -->
        <div class="card full-width">
            <h2>5. Assembled Stanza Canvas</h2>
            <div id="stanza-lines-list" style="display: flex; flex-direction: column; gap: 8px;"></div>
        </div>
    </div>

    <script>
        let currentVersePattern = "10010010";
        let currentPage = 0;
        let currentStanza = [];

        function renderMatrix() {
            const container = document.getElementById('matrix-container');
            container.innerHTML = '';
            currentVersePattern.split('').forEach((bit, idx) => {
                const box = document.createElement('div');
                box.className = `syllable-box stress-${bit}`;
                box.innerHTML = `<div>${bit === '1' ? '⚡' : (bit === '2' ? '⯁' : '•')}</div><div style="font-size:10px; color:#888;">${idx + 1}</div>`;
                box.onclick = () => {
                    let patArr = currentVersePattern.split('');
                    patArr[idx] = patArr[idx] === '0' ? '1' : (patArr[idx] === '1' ? '2' : '0');
                    currentVersePattern = patArr.join('');
                    document.getElementById('target-stress').value = currentVersePattern;
                    renderMatrix();
                    fetchLineCandidates(0);
                };
                container.appendChild(box);
            });
        }

        async function fetchPolygons() {
            const n = document.getElementById('n-input').value;
            const f = document.getElementById('poly-filter').value;
            const res = await fetch(`/api/polygons?n=${n}&filter=${f}`);
            const data = await res.json();
            const list = document.getElementById('polygon-list');
            list.innerHTML = '';
            
            data.polygons.forEach(p => {
                const el = document.createElement('div');
                el.className = 'poly-item';
                el.innerHTML = `<span><strong>[Fwd: ${p.pattern_str}]</strong> <span style="color:#aaa;">[Rev: ${p.reverse_str}]</span></span> <span style="color:#ffd54f;">${p.label}</span>`;
                el.onclick = () => {
                    currentVersePattern = p.pattern_str;
                    document.getElementById('target-stress').value = currentVersePattern;
                    renderMatrix();
                    fetchLineCandidates(0);
                };
                list.appendChild(el);
            });
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
                currentVersePattern = data.full_stress;
                document.getElementById('target-stress').value = currentVersePattern;
                renderMatrix();
                fetchLineCandidates(0);
            }
        }

        async function fetchLineCandidates(page) {
            currentPage = page;
            document.getElementById('page-display').innerText = `Batch ${currentPage + 1}`;
            const stress = document.getElementById('target-stress').value;
            const tail = document.getElementById('context-tail').value;

            const res = await fetch('/api/full_line_stepper', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ target_stress: stress, context_tail: tail, page: currentPage })
            });

            const data = await res.json();
            const container = document.getElementById('full-line-results');
            container.innerHTML = '';

            if(!data.candidates || data.candidates.length === 0) {
                container.innerHTML = '<p style="color: #888;">No lines match this exact configuration. Try expanding/contracting span or changing batch.</p>';
                return;
            }

            data.candidates.forEach(item => {
                const el = document.createElement('div');
                el.className = 'line-candidate-box';
                el.innerHTML = `
                    <div>
                        <strong style="font-size: 15px;">${item.line}</strong>
                        <div style="margin-top: 4px;">
                            <span class="badge">Meter: ${item.stress}</span>
                            <span class="badge" style="color: #4db6ac;">Grammar: ${Math.round(item.grammar_score * 100)}%</span>
                            <span style="font-size: 11px; color: #ffd54f;">Assonance: +${item.internal_rhyme_bonus}</span>
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
            fetchLineCandidates(0);
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
                    <span class="badge">${item.stress}</span>
                `;
                container.appendChild(el);
            });
        }

        function changePage(delta) {
            const newPage = Math.max(0, currentPage + delta);
            fetchLineCandidates(newPage);
        }

        function autoExpandSpan(delta) {
            let val = document.getElementById('target-stress').value;
            if(delta > 0) val += '0';
            else if(val.length > 1) val = val.slice(0, -1);
            currentVersePattern = val;
            document.getElementById('target-stress').value = val;
            renderMatrix();
            fetchLineCandidates(0);
        }

        renderMatrix();
        fetchPolygons();
        fetchLineCandidates(0);
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    print("Running Unified Studio on http://0.0.0.0:5009")
    app.run(host="0.0.0.0", port=5009, debug=True)
