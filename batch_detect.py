import csv
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from google.cloud import vision
from google.cloud import translate_v2 as translate

# ==========================================
# ファイルパスの設定
# ==========================================
IMAGE_DIR = Path("test_image")
RESULT_JSON = Path("results_v9_logo_only.json")
TRANS_CACHE_JSON = Path("translation_cache.json")
LABELS_CSV = Path("labels_v17_hybrid_ensemble.csv") # ハイブリッド版
KOTO_CSV = Path("131083_028_garbage_separation.csv")
COLORS_CSV = Path("colors.csv")
C01_CSV = Path("C01.csv")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# 英語の素材タグを日本語の7大素材に強制変換するルールブック
MATERIAL_LABELS = {
    "paper": "紙類", "paper product": "紙類", "cardboard": "段ボール・厚紙",
    "wood": "木製品・木くず", 
    "textile": "布類・衣類", "woolen": "布類・衣類", "wool": "布類・衣類", "knitting": "布類・衣類", "woven fabric": "布類・衣類", "linens": "布類・シーツ類",
    "plastic": "プラスチック製品", 
    "metal": "金属製品", 
    "glass": "ガラス製品・びん"
}

# NG概念のリスト（類似度が高い場合は除外する）
NG_CONCEPT_ANCHORS = [
    "ceiling wall floor room architecture building interior lighting",
    "sky cloud weather nature landscape geographical",
    "liquid fluid water droplet solution chemical",
    "font typography graphic design logo brand label packaging barcode qr code",
]

# 指定されたパスのJSONファイルを読み込む関数
def load_json(path):
    # ファイルが存在する場合はファイルを読み込み、辞書型データに変換する
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    # ファイルが存在しない場合は空の辞書を返す
    return {}

# 辞書型データを指定されたパスのJSONファイルに保存する関数
def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

# Google Cloud Vision APIに画像を送り、ラベル検出、物体検出、ウェブ検出、ロゴ検出を行う関数
def detect_labels_vision(path):
    # Google Cloud Vision APIのクライアントを作成
    client = vision.ImageAnnotatorClient()
    # 画像ファイルをバイナリとして読み込む
    content = path.read_bytes()
    # Google Cloud Vision APIに送る専用の画像データ形式にセット
    image = vision.Image(content=content)
    # AIに頼むラベルタスクの種類を指定(ラベル、物体、ウェブ、ロゴ)
    features = [
        vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION, max_results=10),
        vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION, max_results=5),
        vision.Feature(type_=vision.Feature.Type.WEB_DETECTION, max_results=5),
        vision.Feature(type_=vision.Feature.Type.LOGO_DETECTION, max_results=3),
    ]
    # 画像とタスクをセットにしてリクエストを作成
    request = vision.AnnotateImageRequest(image=image, features=features)
    # Google Cloud Vision APIにリクエストを送信してレスポンスを受け取る
    response = client.annotate_image(request=request)
    if response.error.message:
        raise Exception(f"Vision API Error: {response.error.message}")

    # 複数タスクの結果を統合先
    merged = {}
    # ラベル検出の結果を統合
    for lb in response.label_annotations:
        merged[lb.description] = round(lb.score, 4)
    # 物体検出の結果を統合（同じ名前の物体があればスコアが高い方を採用）
    for obj in response.localized_object_annotations:
        score = round(obj.score, 4)
        if obj.name not in merged or score > merged[obj.name]:
            merged[obj.name] = score
    # ウェブ検出の結果を統合（スコアが0.65以上のものだけ採用）
    if response.web_detection and response.web_detection.web_entities:
        for ent in response.web_detection.web_entities:
            if ent.description and ent.score >= 0.65:
                norm_score = round(min(ent.score, 0.95), 4)
                if ent.description not in merged:
                    merged[ent.description] = norm_score
    # スコアの高い順にソートしてリスト化
    sorted_tags = [{"description": k, "score": v} for k, v in sorted(merged.items(), key=lambda x: x[1], reverse=True)]
    # ロゴ検出の結果をリスト化
    logos = [logo.description for logo in response.logo_annotations]
    # 最終的な結果を辞書型で返す
    return {"tags": sorted_tags, "logos": logos, "texts": []}

# 除外ワードセットをCSVから読み込む関数
def load_exclusion_sets():
    # 重複を避けるためにセットを作成
    exclude_words = set()
    # カラー系csvファイルを使いやすいように読み込む
    if COLORS_CSV.exists():
        with COLORS_CSV.open("r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("name"):
                    exclude_words.add(row["name"].strip().lower())
    # 行政機関・組織名・地名csvファイルを使いやすいように読み込む
    if C01_CSV.exists():
        with C01_CSV.open("r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                val = row.get("OriginatingGeneratingCentres_en")
                if val:
                    exclude_words.add(val.strip().lower())
    return exclude_words # カラー、概念系を除外するためのセット

# 江東区のルールをAIが検索できるようにする準備
def load_koto_db(model):
    koto_items = []
    # 江東区のCSVを読み込み、品目・分別区分・備考を抽出
    with KOTO_CSV.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            koto_items.append({
                "品目": row["ゴミの品目"],
                "分別区分": row["分別区分"],
                "備考": row["備考"]
            })
    print(f"江東区データ {len(koto_items)} 件をベクトル化しています...")
    # 品目名だけを抽出してベクトル化
    item_names = [item["品目"] for item in koto_items]
    # 品目名のリストを一気にベクトル化(384次元)
    embeddings = model.encode(item_names, normalize_embeddings=True)
    return koto_items, embeddings # 品目リストとベクトルを返す

# 江東区のルールを英語と日本語のベクトル平均化を用いて検索
# 英語クエリ、日本語クエリ、AIモデル、江東区データ、江東区ベクトル、上位k件を返す
def search_koto_rules_hybrid(query_en, query_ja, model, koto_items, koto_embeddings, top_k=3):
    # 英語と日本語、両方のベクトルを作る
    vec_en = model.encode([query_en], normalize_embeddings=True) # 英語版
    vec_ja = model.encode([query_ja], normalize_embeddings=True) # 日本語版
    
    # 江東区の品目ベクトルとクエリベクトルの内積を計算して類似度を求める
    sim_en = np.dot(koto_embeddings, vec_en.T).squeeze() # 英語の類似度
    sim_ja = np.dot(koto_embeddings, vec_ja.T).squeeze() # 日本語の類似度
    
    # 英語と日本語のスコアリストを平均化してブレンドする
    blended_sim = (sim_en + sim_ja) / 2.0

    # スコアが高い上位k件のインデックス番号を取得する
    top_indices = np.argsort(blended_sim)[::-1][:top_k]

    # 上位k件の品目名、分別区分、備考、類似度を返す
    return [{
        "品目": koto_items[idx]["品目"],
        "分別区分": koto_items[idx]["分別区分"],
        "備考": koto_items[idx]["備考"],
        "類似度": round(float(blended_sim[idx]), 4)
    } for idx in top_indices]

# 表示用の警告アラート分
def extract_parts_alerts(memo):
    # メモに何もない場合は「-」を返す
    if not memo or not isinstance(memo, str):
        return "-"
    alerts = [] # 警告アラートを格納するリスト
    # メモの内容に応じて警告アラートを追加
    if "キャップ" in memo or "ラベル" in memo:
        alerts.append("⚠️ キャップとラベルを外す")
    if "汚れ" in memo or "すす" in memo or "洗っ" in memo:
        alerts.append("⚠️ 汚れを落とす（すすぐ）")
    if "電池" in memo:
        alerts.append("⚠️ 電池を抜く")
    if "中身" in memo or "空に" in memo:
        alerts.append("⚠️ 中身を使い切る")
    if "30cm" in memo:
        alerts.append("⚠️ 30cm以上は粗大ごみ")
    return " / ".join(alerts) if alerts else "特になし"

# Google Cloudを使って英語を日本語に一気に翻訳
def pre_translate_all_tags(results, trans_cache):
    # 翻訳するための英単語のセットを作成
    needed_translations = set()
    # results辞書(各画像のタグ情報のまとまり)から英語のタグを取り出す
    for data in results.values():
        # dataが辞書型の場合は"tags"キーの値を取得、リスト型の場合はそのまま使用
        labels = data.get("tags", []) if isinstance(data, dict) else data
        for lb in labels:
            # "tags"の中から"description"キーの値を取り出す
            desc = lb["description"]
            # 日本語の文字が含まれてないかつ翻訳キャッシュにない場合は翻訳待ちセットに追加
            if not any("\u3000" <= ch <= "\u9fff" for ch in desc) and desc not in trans_cache:
                needed_translations.add(desc)

    # 翻訳待ちリストがある場合はtranslateに一括で投げる
    if needed_translations:
        # Google Cloud APIに送るためにセットをリスト型に変換する
        needed_list = list(needed_translations)
        # 何件翻訳するのか
        print(f"\n新規タグ {len(needed_list)} 件をGoogle Cloud公式APIで一括翻訳します...")
        try:
            # Google Cloud Translation APIのクライアントを作成
            translate_client = translate.Client()
            # 翻訳を実行し、結果をresultで受け取る
            result = translate_client.translate(needed_list, target_language="ja")
            # 翻訳部分
            for en, res in zip(needed_list, result):
                ja = res["translatedText"]
                # 翻訳結果をキャッシュに保存
                trans_cache[en] = ja
                print(f"  [公式翻訳完了] {en} -> {ja}")
        except Exception as e: # APIエラーが発生した場合
            print(f"公式翻訳APIエラー: {e}")

# 辞書から翻訳結果を取り出す
def auto_translate_en_to_ja(text, trans_cache):
    if any("\u3000" <= ch <= "\u9fff" for ch in text):
        return text
    return trans_cache.get(text, text)

# これまでの関数を組み合わせてゴミの最終判別
def export_csv(results, model, koto_items, koto_embeddings, exclude_words, trans_cache):
    # しきい値
    MAX_TAG_RANK = 6 # 1つに対するタグの最大個数
    VISION_CONF_MIN = 0.55 # Vision APIの信頼度による足切りライン
    COMBINED_SCORE_THRESHOLD = 0.65 # 断定のための合格ライン
    VECTOR_SIM_MIN_THRESHOLD = 0.75 # ベクトル類似度の足切りライン
    CANDIDATE_SIM_THRESHOLD = 0.55 # 候補から選択のライン
    NG_ANCHOR_THRESHOLD = 0.58 # タグ除外の基準
    MATERIAL_PRIORITY_MARGIN = 0.15 # 素材のタグ優先補正度
    # NGワードのベクトルを事前に作成しておく
    ng_embeddings = model.encode(NG_CONCEPT_ANCHORS, normalize_embeddings=True)

    # CSVファイルに結果を書き込む
    with LABELS_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        # CSVのヘッダー行を書き込む
        writer.writerow([ 
            "file", "判定ステータス", 
            "【1. ゴミの種類】", "【2. 品目名】", "【3. 検出素材】", "【4. ブランド】", "【5. パーツ注意点】",
            "江東区_備考", "代替・別選択肢", "総合スコア", "最良採用クエリ"
        ])

        # 画像1件ごとに処理を行う
        for name, data in sorted(results.items()):
            # リストか辞書か見分ける
            if isinstance(data, list):
                labels = data
                brand_str = "-"
            else: # 辞書の場合はタグをlabels、ロゴを文字列としてbrand_strで繋ぐ(ロゴがない場合は"-"を使う)
                labels = data.get("tags", [])
                logos = data.get("logos", [])
                brand_str = " ".join(logos) if logos else "-"
            # タグが見つからない場合は「検出物なし」とする
            if not labels:
                writer.writerow([name, "検出物なし", "-", "-", "-", brand_str, "-", "-", "-", 0, "-"])
                continue

            specific_tags = [] # 本物のゴミタグ
            material_tags = [] # 素材のタグ
            
            for lb in labels:
                desc = lb["description"]
                desc_lower = desc.lower().strip() # 英単語を小文字+空白除去で取り出す
                score = lb["score"] # Google Vision APIの自信度

                # 自信度が0.55(VISION_CONF_MIN)未満なら信用できないためスルー
                if score < VISION_CONF_MIN: 
                    continue
                # 英単語がNGワードリストに引っかかる場合はスルー
                if desc_lower in exclude_words: 
                    continue
                # 英単語が素材の場合は素材リストに入れる
                if desc_lower in MATERIAL_LABELS: 
                    material_tags.append(lb)
                    continue

                # 残った単語をベクトル化
                tag_vec = model.encode([desc_lower], normalize_embeddings=True)
                # NG概念ベクトルと内積を計算し、最大値を代入(NGワードに似てしまっているため)
                ng_sim = float(np.max(np.dot(ng_embeddings, tag_vec.T)))
                # NG類似度が0.58(NG_ANCHOR_THRESHOLD)以上であればその単語を除外
                if ng_sim >= NG_ANCHOR_THRESHOLD:
                    continue
                # フィルターを通過したタグを格納
                specific_tags.append(lb)
            # 素材タグを翻訳、重複なしかつ"/"で繋いだ文字列で保管
            detected_materials = []
            for m in material_tags:
                m_jp = MATERIAL_LABELS.get(m["description"].lower().strip())
                if m_jp and m_jp not in detected_materials:
                    detected_materials.append(m_jp)
            mat_str = " / ".join(detected_materials) if detected_materials else "不明"

            best_candidate = None # 一番スコアが高かった江東区のゴミ情報
            # フィルターを通過した上位6つ(MAX_TAG_RANK)のタグを集める
            if specific_tags:
                top_n_tags = specific_tags[:MAX_TAG_RANK]
                trials = []
                
                # 検索するためのワードリストを詰める
                for idx, t in enumerate(top_n_tags, 1):
                    en_text = t["description"]
                    ja_text = auto_translate_en_to_ja(en_text, trans_cache)
                    trials.append((en_text, ja_text, t["score"], f"{en_text} ✕ {ja_text} (ハイブリッド)"))

                # 上位3つのタグをスペースで繋げる、平均値をtrialsに追加する
                top3_for_comb = top_n_tags[:3]
                if len(top3_for_comb) > 1:
                    comb_en = " ".join([t["description"] for t in top3_for_comb])
                    comb_ja = " ".join([auto_translate_en_to_ja(t["description"], trans_cache) for t in top3_for_comb])
                    avg_v_score = round(float(np.mean([t["score"] for t in top3_for_comb])), 4)
                    trials.append((comb_en, comb_ja, avg_v_score, "上位合体 (ハイブリッド)"))

                for q_en, q_ja, v_sc, q_label in trials:
                    # 英語と日本語のクエリ両方から江東区データでのスコアが高い上位3件を受け取る
                    m_list = search_koto_rules_hybrid(q_en, q_ja, model, koto_items, koto_embeddings, top_k=3)
                    # 類似度1位をVisionスコアと掛け算して総合スコアとする
                    top_sim = m_list[0]["類似度"]
                    comb_sc = round(v_sc * top_sim, 4)

                    # 一番スコアが高かった江東区のゴミ情報が空もしくは総合スコアが上回っていた場合は情報を書き換える
                    if best_candidate is None or comb_sc > best_candidate["combined"]:
                        best_candidate = {
                            "query_label": q_label,
                            "v_score": v_sc,
                            "matches": m_list,
                            "sim": top_sim,
                            "combined": comb_sc,
                        }

            # 品目タグよりも素材タグの自信度が0.15(MATERIAL_PRIORITY_MARGIN)以上高い場合は素材から判定した方が良いとフラグを立てる
            prefer_material = False
            if material_tags and best_candidate:
                top_mat_score = material_tags[0]["score"]
                if (top_mat_score - best_candidate["v_score"] >= MATERIAL_PRIORITY_MARGIN):
                    prefer_material = True
            
            # 江東区DBと類似度0.88以上かつ総合スコア05以上の場合は素材より品目判定を強行突破する
            special_match = False
            if best_candidate and best_candidate["sim"] >= 0.88 and best_candidate["combined"] >= 0.50:
                prefer_material = False
                special_match = True

            # 素材優先でない、類似度0.75以上、総合スコアが0.65以上の3つ条件を満たしてる場合もしくは、上の特定フラグ(special_match)を立っている場合
            if (best_candidate and not prefer_material) and (
                (best_candidate["sim"] >= VECTOR_SIM_MIN_THRESHOLD and best_candidate["combined"] >= COMBINED_SCORE_THRESHOLD)
                or special_match
            ):
                # 1位のデータをbestに、2位、3位の品目名を繋げたaltsを作る
                best = best_candidate["matches"][0]
                parts_alert = extract_parts_alerts(best["備考"])
                alts = f"2位: {best_candidate['matches'][1]['品目']} / 3位: {best_candidate['matches'][2]['品目']}"
                # CSVの1行に情報を書き込む
                writer.writerow([
                    name, "確定（マッチ）", 
                    best["分別区分"], best["品目"], mat_str, brand_str, parts_alert,
                    best["備考"], alts, best_candidate["combined"], best_candidate["query_label"]
                ])
                continue

            # 断定が難しいが素材タグがある場合、素材選択としてCSVに書き込む
            if material_tags:
                mat_jp = detected_materials[0] if detected_materials else "素材"
                fallback = "-"
                if best_candidate:
                    fallback = f"参考1位: {best_candidate['matches'][0]['品目']} (類似度:{best_candidate['matches'][0]['類似度']})"
                writer.writerow([
                    name, "素材から選択", 
                    "各素材の分別を確認", mat_jp, mat_str, brand_str, "素材別の分解ルールを確認",
                    "-", fallback, "-", "-"
                ])
                continue

            # 確定にもならず、素材タグもなしだが、類似度が0.55(CANDIDATE_SIM_THRESHOLD)以上ある場合は候補から選択としてCSVに書き込む
            if best_candidate and best_candidate["sim"] >= CANDIDATE_SIM_THRESHOLD:
                best = best_candidate["matches"][0]
                parts_alert = extract_parts_alerts(best["備考"])
                alts = f"候補2: {best_candidate['matches'][1]['品目']} / 候補3: {best_candidate['matches'][2]['品目']}"
                writer.writerow([
                    name, "候補から選択", 
                    best["分別区分"], f"候補: {best['品目']}", mat_str, brand_str, parts_alert,
                    best["備考"], alts, best_candidate["combined"], best_candidate["query_label"]
                ])
                continue

            # 全ての条件に当てはまらない場合は特定不能と書き込む
            writer.writerow([
                name, "特定不能（足切り）", 
                "不明", "不明", mat_str, brand_str, "-",
                "-", "-", 0.0, "-"
            ])

def main():
    # 画像ファイルを集めてアルファベット順に並べる
    images = sorted(p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in EXTENSIONS)
    # 過去にAPI通信した結果をJSONから読み込む
    results = load_json(RESULT_JSON)
    trans_cache = load_json(TRANS_CACHE_JSON)
    # JSONに結果が保存されてない画像を取り出す
    todo = [p for p in images if p.name not in results]

    # 新しい画像があれば、Vision APIに送り、JSONファイルに上書き保存する
    if todo:
        for path in todo:
            try:
                results[path.name] = detect_labels_vision(path)
                save_json(RESULT_JSON, results)
            except Exception as e: # エラー時
                print(f"失敗: {path.name} ({e})")
    # 未翻訳の単語があれば一括で翻訳、終わったら翻訳履歴をJSONに保存
    pre_translate_all_tags(results, trans_cache)
    save_json(TRANS_CACHE_JSON, trans_cache)
    # 除外ワードのリスト読み込み
    exclude_words = load_exclusion_sets()
    # ベクトル変換モデルをメモリ上に起動
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    # 江東区DBを読み込み、ベクトル化
    koto_items, koto_embeddings = load_koto_db(model)
    # 準備したデータをゴミ判別関数(export_csv)に渡す
    export_csv(results, model, koto_items, koto_embeddings, exclude_words, trans_cache)
    print(f"完了！ labels_v17_hybrid_ensemble.csv を出力しました。")

if __name__ == "__main__":
    main()
