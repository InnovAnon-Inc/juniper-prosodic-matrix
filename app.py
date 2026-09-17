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

def stress_vector_similarity(target_stress, candidate_stress):
    """Calculates normalized match score between target meter and candidate meter."""
    if not target_stress or not candidate_stress:
        return 0.0
    # Pad or truncate to evaluate match
    min_len = min(len(target_stress), len(candidate_stress))
    max_len = max(len(target_stress), len(candidate_stress))
    
    matches = sum(1 for i in range(min_len) if target_stress[i] == candidate_stress[i])
    # Penalize length discrepancies
    return matches / max_len

# ==============================================================================
# 2. LINE-LEVEL CADENCE & PHONETICS ENGINE
# ==============================================================================
class LineCadenceEngine:
    def __init__(self):
        self.vowels = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
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
        
        # Find rhyme tail (from last primary/secondary stress to end)
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

    def generate_line_candidates(self, target_stress, context_tail=None, pos_sequence=None, limit=10):
        """Assembles whole-line candidates matching target stress & phonetic tails."""
        target_len = len(target_stress)
        if target_len == 0:
            return []

        # Find words that fit chunks of the target stress
        matching_lines = []
        
        # Simple recursive dynamic programming match for line assembly
        def build_phrase(rem_stress, current_phrase, current_tail):
            if len(matching_lines) >= limit * 3:
                return
            if not rem_stress:
                # Calculate whole-line tail distance if context tail exists
                distance = edit_distance(current_tail, context_tail) if context_tail else 0
                matching_lines.append({
                    "line": " ".join([w["word"] for w in current_phrase]),
                    "stress": "".join([w["stress"] for w in current_phrase]),
                    "rhyme_tail": current_tail,
                    "levenshtein_dist": distance
                })
                return

            # Match chunk lengths (1 to 4 syllables)
            for chunk_len in range(1, min(5, len(rem_stress) + 1)):
                sub_stress = rem_stress[:chunk_len]
                # Filter candidates from db matching sub_stress
                candidates = [prof for prof in self.word_db.values() if prof["stress"] == sub_stress]
                for cand in candidates[:3]: # Sample top matches for efficiency
                    build_phrase(
                        rem_stress[chunk_len:], 
                        current_phrase + [cand], 
                        cand["rhyme_tail"]
                    )

        build_phrase(target_stress, [], "")

        # Sort by Levenshtein distance on rhyme tail if context exists
        if context_tail:
            matching_lines.sort(key=lambda x: x["levenshtein_dist"])

        return matching_lines[:limit]

line_engine = LineCadenceEngine()

# ==============================================================================
# 3. BALANCED POLYGON ENGINE
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
    total_vector = sum(cmath.exp(1j * 2 * math.pi * i / N) for i, active in enumerate(pattern) if active)
    cx, cy = (total_vector / sum(pattern)).real, (total_vector / sum(pattern)).imag if sum(pattern) > 0 else (0,0)
    balanced = math.hypot(cx, cy) < 1e-5
    return {
        "pattern_str": "".join(map(str, pattern)),
        "is_balanced": balanced,
        "label": f"{'Cyclotomic Balanced' if balanced else 'Euclidean'} ({sum(pattern)}/{N})"
    }

# ==============================================================================
# 4. API & FLASK ROUTES
# ==============================================================================
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/full_line_stepper", methods=["POST"])
def api_full_line_stepper():
    data = request.get_json() or {}
    target_stress = data.get("target_stress", "")
    context_tail = data.get("context_tail", "")
    
    candidates = line_engine.generate_line_candidates(
        target_stress=target_stress,
        context_tail=context_tail,
        limit=12
    )
    return jsonify({
        "target_stress": target_stress,
        "lines": candidates
    })

# ==============================================================================
# 5. UI TEMPLATE
# ==============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Full-Line Prosodic Stepper & Cadence Engine</title>
    <style>
        body { font-family: 'Segoe UI', monospace, sans-serif; background: #121214; color: #e0e0e0; margin: 0; padding: 20px; }
        h1, h2, h3 { color: #4db6ac; margin-top: 0; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .card { background: #1e1e24; border-radius: 8px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.4); }
        .full-width { grid-column: span 2; }
        input, button, select { background: #2a2a32; border: 1px solid #444; color: #fff; padding: 8px 12px; border-radius: 4px; }
        button { background: #00897b; font-weight: bold; cursor: pointer; }
        button:hover { background: #00bfa5; }
        .button-secondary { background: #37474f; }
        .button-secondary:hover { background: #455a64; }

        .line-candidate-box { 
            background: #18181c; border: 1px solid #333; padding: 12px; margin-bottom: 8px; 
            border-radius: 6px; display: flex; justify-content: space-between; align-items: center; 
            cursor: pointer; transition: border-color 0.2s;
        }
        .line-candidate-box:hover { border-color: #00bfa5; background: #1f2826; }
        .stress-badge { font-family: monospace; color: #00e676; background: #18281e; padding: 2px 6px; border-radius: 4px; font-size: 12px; }
        .dist-badge { font-size: 11px; color: #ffd54f; }
        .stepper-controls { display: flex; gap: 10px; align-items: center; background: #18181c; padding: 10px; border-radius: 6px; margin-bottom: 15px; }
    </style>
</head>
<body>

    <h1>Full-Line Prosodic Stepper & Cadence Engine</h1>

    <div class="grid">
        <div class="card full-width">
            <h2>Line-by-Line Automated Cadence Stepper</h2>
            <p style="font-size: 13px; color: #aaa;">
                Step through entire lines of poetry matched directly against your master stress pattern. 
                Auto-expands/contracts spans and ranks full lines by Levenshtein distance against preceding stanza rhymes.
            </p>

            <div class="stepper-controls">
                <label>Master Line Meter:</label>
                <input type="text" id="target-stress" value="10010010" style="font-family: monospace; width: 140px;">
                <label>Context Rhyme Tail:</label>
                <input type="text" id="context-tail" placeholder="e.g., AH_NG" style="width: 100px;">
                <button onclick="fetchLineCandidates()">Auto-Fit Full Lines ➔</button>
                <button class="button-secondary" onclick="autoExpandSpan(-1)">Contract Span -1</button>
                <button class="button-secondary" onclick="autoExpandSpan(1)">Expand Span +1</button>
            </div>

            <h3>Matching Full-Line Offerings</h3>
            <div id="full-line-results">
                <p style="color: #666;">Click "Auto-Fit Full Lines" to synthesize full-line candidates matching target meter and rhyme.</p>
            </div>
        </div>

        <div class="card full-width">
            <h2>Stanza Canvas</h2>
            <div id="stanza-lines-list" style="display: flex; flex-direction: column; gap: 8px;"></div>
        </div>
    </div>

    <script>
        let currentStanza = [];

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

            if(!data.lines || data.lines.length === 0) {
                container.innerHTML = '<p style="color: #888;">No exact whole-line combinations found for this exact vector. Try expanding or contracting meter span.</p>';
                return;
            }

            data.lines.forEach(item => {
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
            document.getElementById('context-tail').value = tail; // Auto-chain rhyme tail to next line
            renderStanza();
            fetchLineCandidates(); // Refresh next set of options linked to new tail
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

        fetchLineCandidates();
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    print("Running Full-Line Prosodic Stepper on http://0.0.0.0:5009")
    app.run(host="0.0.0.0", port=5009, debug=True)
