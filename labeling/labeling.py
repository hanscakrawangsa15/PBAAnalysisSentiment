"""
labeling.py
============
Web app buat labeling manual komentar (positif / negatif / netral).

- Data mentah (RAW_FILES) TIDAK PERNAH ditimpa/diubah.
- Tiap orang punya file csv SENDIRI-SENDIRI di folder labeling/
  (labels_shafwa.csv, labels_hans.csv, dst) -- jadi kalau dikerjain rame-rame
  lewat git, gak akan ada conflict, soalnya masing-masing cuma nulis ke
  filenya sendiri.
- Sebelum mulai, app nanya dulu "ini siapa?" - pilih salah satu dari 5 nama.
  Jawaban itu nentuin file mana yang dipakai.

Cara kerja:
- Pilih nama dulu di halaman awal
- Komentar ditampilin satu-satu
- Arrow LEFT  (←) = negatif
- Arrow DOWN  (↓) = netral
- Arrow RIGHT (→) = positif
- Arrow UP    (↑) / tombol Previous = balik ke komentar sebelumnya (relabel)
- Auto-save PER KLIK, langsung update + rewrite file csv orang itu

Usage:
    python labeling.py
    Buka browser: http://localhost:5000
"""

from flask import Flask, render_template_string, request, jsonify
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# ============================================================================
# PATH CONFIG - CUSTOMIZE DI SINI
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent

# Daftar semua batch data mentah, akan digabung otomatis
RAW_FILES = [
    BASE_DIR.parent / "data" / "comments.csv",
    BASE_DIR.parent / "data" / "comments_batch2.csv"
]
OUTPUT_DIR = BASE_DIR                                 # tiap orang punya file sendiri di sini

LABELERS = ["Shafwa", "Hans", "Gerald", "Clay", "Baqhiz"]
LABEL_CHOICES = ["negatif", "netral", "positif"]

# ============================================================================
# STATE
# ============================================================================

state = {
    'dfs': {},       # nama -> dataframe punya orang itu
    'sessions': {},  # nama -> {'queue': [list of df index], 'pos': int}
}

# ============================================================================
# LOAD / SAVE (per orang, file terpisah)
# ============================================================================

def output_path(name):
    return OUTPUT_DIR / f"labels_{name.lower()}.csv"

def load_person_df(name):
    """Load file csv punya 1 orang. Kalau ada batch baru, otomatis ditambahkan di bawahnya."""
    if name in state['dfs']:
        return state['dfs'][name]

    path = output_path(name)
    
    # Gabungkan semua data mentah
    raw_dfs = []
    for rf in RAW_FILES:
        if rf.exists():
            raw_dfs.append(pd.read_csv(rf, dtype=str))
            
    if not raw_dfs:
        raise FileNotFoundError("Tidak ada file data mentah yang ditemukan di folder data!")
        
    master_df = pd.concat(raw_dfs, ignore_index=True)

    # Cek file output user
    if path.exists():
        df = pd.read_csv(path, dtype=str)
        # Jika master_df lebih panjang, berarti ada batch baru. Append sisanya.
        if len(master_df) > len(df):
            new_data = master_df.iloc[len(df):].copy()
            new_data['label'] = ''
            df = pd.concat([df, new_data], ignore_index=True)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False, encoding='utf-8')
    else:
        df = master_df.copy()
        df['label'] = ''
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding='utf-8')

    df['comment_text'] = df['comment_text'].fillna('')
    if 'label' not in df.columns:
        df['label'] = ''
    df['label'] = df['label'].fillna('')

    state['dfs'][name] = df
    return df

def save_person_df(name):
    state['dfs'][name].to_csv(output_path(name), index=False, encoding='utf-8')

def get_session(name):
    """Ambil (atau bikin) antrian & posisi buat 1 orang."""
    if name not in state['sessions']:
        df = load_person_df(name)
        pending = df[df['label'] == ''].index.tolist()
        state['sessions'][name] = {'queue': pending, 'pos': 0}
    return state['sessions'][name]

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    name = request.args.get('labeler', '')

    if name not in LABELERS:
        return render_template_string(SELECT_HTML, labelers=LABELERS)

    try:
        df = load_person_df(name)
        sess = get_session(name)
        queue, pos = sess['queue'], sess['pos']

        total = len(queue)
        reviewed = sum(1 for idx in queue if df.at[idx, 'label'] != '')

        if not queue or pos >= total:
            return render_template_string(DONE_HTML, name=name, total=total)

        df_idx = queue[pos]
        row = df.loc[df_idx]

        return render_template_string(
            MAIN_HTML,
            name=name,
            progress=(reviewed / total * 100) if total else 0,
            progress_bar=f"{reviewed}/{total}",
            username=row['username'],
            comment_text=str(row['comment_text']),
            likes_count=row['likes_count'],
            post_url=row['post_url'],
            has_prev=(pos > 0),
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {e}", 500

@app.route('/label', methods=['POST'])
def label_comment():
    try:
        name = request.args.get('labeler', '')
        if name not in LABELERS:
            return jsonify({'error': 'labeler tidak valid'}), 400

        payload = request.get_json(silent=True)
        label = payload.get('label') if payload else None
        if label not in LABEL_CHOICES:
            return jsonify({'error': f'Invalid label: {label}'}), 400

        df = load_person_df(name)
        sess = get_session(name)
        queue, pos = sess['queue'], sess['pos']

        if pos >= len(queue):
            return jsonify({'error': 'Semua komentar sudah dilabel'}), 400

        df_idx = queue[pos]
        df.at[df_idx, 'label'] = label
        save_person_df(name)

        sess['pos'] += 1

        if sess['pos'] >= len(queue):
            return jsonify({'done': True, 'message': f'✅ {name} sudah selesai labeling {len(queue)} komentar!'})

        return jsonify({'success': True})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/previous', methods=['POST'])
def previous_comment():
    try:
        name = request.args.get('labeler', '')
        if name not in LABELERS:
            return jsonify({'error': 'labeler tidak valid'}), 400

        sess = get_session(name)
        if sess['pos'] <= 0:
            return jsonify({'error': 'Sudah di komentar paling awal'}), 400
        sess['pos'] -= 1
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# HTML
# ============================================================================

SELECT_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Labeling Komentar</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; display: flex; justify-content: center; align-items: center;
            padding: 30px 16px;
        }
        .container {
            background: white; border-radius: 14px; box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 380px; width: 100%; padding: 32px 26px; text-align: center;
        }
        h1 { font-size: 20px; margin-bottom: 6px; }
        p { font-size: 14px; color: #666; margin-bottom: 22px; }
        a.pick {
            display: block; padding: 16px; margin-bottom: 12px; border-radius: 10px;
            background: #f0f0f5; color: #333; text-decoration: none; font-weight: 700;
            font-size: 16px;
        }
        a.pick:hover { background: #667eea; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Labeling Komentar</h1>
        <p>Ini siapa yang labeling?</p>
        {% for l in labelers %}
        <a class="pick" href="/?labeler={{ l }}">{{ l }}</a>
        {% endfor %}
    </div>
</body>
</html>
"""

MAIN_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Labeling Komentar - {{ name }}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; display: flex; justify-content: center; align-items: flex-start;
            padding: 30px 16px;
        }
        .container {
            background: white; border-radius: 14px; box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 480px; width: 100%; padding: 26px 24px;
        }
        .top-bar { display: flex; align-items: center; gap: 14px; margin-bottom: 6px; }
        .progress { flex: 1; }
        .progress-bar { width: 100%; height: 8px; background: #e0e0e0; border-radius: 10px; overflow: hidden; margin-bottom: 8px; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); width: {{ progress }}%; }
        .progress-text { font-size: 14px; color: #666; font-weight: 500; }
        .btn-prev {
            flex-shrink: 0; background: #f0f0f5; color: #555; border: none;
            border-radius: 8px; padding: 10px 16px; font-size: 14px; font-weight: 600; cursor: pointer;
        }
        .btn-prev:disabled { opacity: 0.35; cursor: not-allowed; }
        .who { font-size: 12px; color: #999; text-align: right; margin-bottom: 14px; }
        .who a { color: #667eea; text-decoration: none; }

        .username { font-size: 18px; color: #333; margin-bottom: 10px; font-weight: 700; }
        .comment-text {
            font-size: 18px; color: #333; background: #f9f9f9; padding: 18px;
            border-radius: 10px; line-height: 1.6; margin-bottom: 14px;
            white-space: pre-wrap; word-break: break-word;
        }
        .meta { font-size: 13px; color: #999; margin-bottom: 20px; }
        .meta a { color: #667eea; text-decoration: none; }

        .controls { display: flex; flex-direction: column; gap: 12px; }
        .btn { padding: 18px 16px; border: none; border-radius: 10px; font-size: 17px; font-weight: 700; cursor: pointer; color: white; }
        .btn-negatif { background: #ff6b6b; }
        .btn-negatif:hover { background: #ff5252; }
        .btn-netral { background: #868e96; }
        .btn-netral:hover { background: #74797f; }
        .btn-positif { background: #51cf66; }
        .btn-positif:hover { background: #40c057; }

        .hint { margin-top: 16px; font-size: 13px; color: #999; text-align: center; }
        .loading { opacity: 0.5; pointer-events: none; }
    </style>
</head>
<body>
    <div class="container">
        <div class="top-bar">
            <button class="btn-prev" id="prevBtn" onclick="goPrevious()" {{ 'disabled' if not has_prev else '' }}>
                ↑ Previous
            </button>
            <div class="progress">
                <div class="progress-bar"><div class="progress-fill"></div></div>
                <div class="progress-text">{{ progress_bar }} dilabel</div>
            </div>
        </div>
        <div class="who">{{ name }} &nbsp;·&nbsp; <a href="/">ganti orang</a></div>

        <div class="username">@{{ username }}</div>
        <div class="comment-text">{{ comment_text }}</div>
        <div class="meta">
            ❤️ {{ likes_count }} &nbsp;|&nbsp;
            <a href="{{ post_url }}" target="_blank">buka post →</a>
        </div>

        <div class="controls">
            <button class="btn btn-negatif" onclick="labelComment('negatif')">← Negatif</button>
            <button class="btn btn-netral" onclick="labelComment('netral')">↓ Netral</button>
            <button class="btn btn-positif" onclick="labelComment('positif')">Positif →</button>
        </div>

        <div class="hint">💡 ← Negatif &nbsp;|&nbsp; ↓ Netral &nbsp;|&nbsp; → Positif &nbsp;|&nbsp; ↑ Previous</div>
    </div>

    <script>
        const LABELER = {{ name | tojson }};

        function setLoading(state) {
            document.querySelectorAll('.btn, .btn-prev').forEach(b => {
                if (state) b.classList.add('loading'); else b.classList.remove('loading');
            });
        }

        function labelComment(label) {
            setLoading(true);
            fetch('/label?labeler=' + encodeURIComponent(LABELER), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ label: label })
            })
            .then(r => r.json().catch(() => { throw new Error('Server error, cek terminal Flask.'); }))
            .then(data => {
                if (data.error) { alert('Error: ' + data.error); setLoading(false); return; }
                if (data.done) {
                    document.body.innerHTML = '<div class="container" style="text-align:center;margin-top:50px;max-width:480px;"><h1 style="color:#51cf66;font-size:48px;">✅</h1><h2>' + data.message + '</h2></div>';
                } else if (data.success) { location.reload(); }
            })
            .catch(e => { alert('Error: ' + e.message); setLoading(false); });
        }

        function goPrevious() {
            setLoading(true);
            fetch('/previous?labeler=' + encodeURIComponent(LABELER), { method: 'POST' })
            .then(r => r.json().catch(() => { throw new Error('Server error.'); }))
            .then(data => {
                if (data.error) { setLoading(false); return; }
                if (data.success) location.reload();
            })
            .catch(e => { alert('Error: ' + e.message); setLoading(false); });
        }

        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') labelComment('negatif');
            else if (e.key === 'ArrowDown') labelComment('netral');
            else if (e.key === 'ArrowRight') labelComment('positif');
            else if (e.key === 'ArrowUp') { e.preventDefault(); goPrevious(); }
        });
    </script>
</body>
</html>
"""

DONE_HTML = """
<!DOCTYPE html>
<html><head><title>Done!</title>
<style>
    * { margin:0; padding:0; }
    body { font-family:-apple-system,sans-serif; background:linear-gradient(135deg,#667eea,#764ba2);
        min-height:100vh; display:flex; justify-content:center; align-items:center; padding:20px; }
    .container { background:white; border-radius:12px; box-shadow:0 20px 60px rgba(0,0,0,0.3);
        max-width:480px; width:100%; padding:50px; text-align:center; }
    h1 { font-size:72px; margin-bottom:20px; }
    h2 { font-size:22px; color:#333; margin-bottom:15px; }
    a { color: #667eea; }
</style></head>
<body><div class="container">
    <h1>✅</h1>
    <h2>{{ name }}, semua {{ total }} komentar udah dilabel!</h2>
    <p><a href="/">← ganti orang</a></p>
</div></body></html>
"""

# ============================================================================
# RUN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("LABELING - LABELING MANUAL KOMENTAR (1 file csv per orang)")
    print("="*70)

    print(f"\n🌐 Buka browser: http://localhost:5000")
    print(f"⌨️  ← Negatif | ↓ Netral | → Positif | ↑ Previous\n")

    app.run(debug=False, host='0.0.0.0', port=5000)