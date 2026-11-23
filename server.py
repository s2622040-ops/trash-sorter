import csv
import json
from pathlib import Path
from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from google.cloud import vision
from google.cloud import translate_v2 as translate

# ファイルパス（本番で必要なものだけ残す）
TRANS_CACHE_JSON = Path("translation_cache.json")
KOTO_CSV = Path("131083_028_garbage_separation.csv")
COLORS_CSV = Path("colors.csv")
C01_CSV = Path("C01.csv")

MATERIAL_LABELS = {
    "paper": "紙類", "paper product": "紙類", "cardboard": "段ボール・厚紙",
    "wood": "木製品・木くず",
    "textile": "布類・衣類", "woolen": "布類・衣類", "wool": "布類・衣類",
    "knitting": "布類・衣類", "woven fabric": "布類・衣類", "linens": "布類・シーツ類",
    "plastic": "プラスチック製品",
    "metal": "金属製品",
    "glass": "ガラス製品・びん"
}

NG_CONCEPT_ANCHORS = [
    "ceiling wall floor room architecture building interior lighting",
    "sky cloud weather nature landscape geographical",
    "liquid fluid water droplet solution chemical",
    "font typography graphic design logo brand label packaging barcode qr code",
]

# サーバー起動時にAIモデルや江東区DBのデータを入れるためのボックス
state = {}
# ファイルが存在すれば読み込む
def load_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
# 辞書データを、日本語が文字化けしない形式に整形してJSONファイルに保存
def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
# 色名と行政機関・組織名を読み込み、重複のない除外ワードセットを作る
def load_exclusion_sets():
    exclude_words = set()
    if COLORS_CSV.exists():
        with COLORS_CSV.open("r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("name"):
                    exclude_words.add(row["name"].strip().lower())
    if C01_CSV.exists():
        with C01_CSV.open("r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                val = row.get("OriginatingGeneratingCentres_en")
                if val:
                    exclude_words.add(val.strip().lower())
    return exclude_words

# サーバー起動時に1回だけ実行される初期化処理
@asynccontextmanager
async def lifespan(app: FastAPI):
    # AIモデルをメモリに広げる
    print("AIモデルと江東区DBをメモリに読み込み中...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    # 江東区CSVを読み込み、品目名のリストを作成
    koto_items = []
    with KOTO_CSV.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            koto_items.append({
                "品目": row["ゴミの品目"],
                "分別区分": row["分別区分"],
                "備考": row["備考"]
            })
    item_names = [item["品目"] for item in koto_items]

    # 共有ボックス(state)に格納する
    state["model"] = model # AIモデル
    state["koto_items"] = koto_items # 江東区データ
    state["koto_embeddings"] = model.encode(item_names, normalize_embeddings=True) # 江東区ベクトル
    state["ng_embeddings"] = model.encode(NG_CONCEPT_ANCHORS, normalize_embeddings=True) # NGベクトル
    state["exclude_words"] = load_exclusion_sets() # 除外ワード
    state["trans_cache"] = load_json(TRANS_CACHE_JSON) # 翻訳キャッシュ
    # 起動準備完了、サーバーの受付開始
    print("サーバー準備完了！")
    yield
# サーバー初期化処理(lifespan)を組み込んだFastAPIサーバーを本体に作成
app = FastAPI(lifespan=lifespan)

# どのアプリ、通信方法、ヘッダー情報でも受け入れ可能にする
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 画像解析
def detect_labels_vision_bytes(content: bytes):
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=content)
    features = [
        vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION, max_results=10),
        vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION, max_results=5),
        vision.Feature(type_=vision.Feature.Type.WEB_DETECTION, max_results=5),
        vision.Feature(type_=vision.Feature.Type.LOGO_DETECTION, max_results=3),
    ]
    response = client.annotate_image(request=vision.AnnotateImageRequest(image=image, features=features))
    if response.error.message:
        raise Exception(f"Vision API Error: {response.error.message}")

    merged = {}
    for lb in response.label_annotations:
        merged[lb.description] = round(lb.score, 4)
    for obj in response.localized_object_annotations:
        score = round(obj.score, 4)
        if obj.name not in merged or score > merged[obj.name]:
            merged[obj.name] = score
    if response.web_detection and response.web_detection.web_entities:
        for ent in response.web_detection.web_entities:
            if ent.description and ent.score >= 0.65:
                norm_score = round(min(ent.score, 0.95), 4)
                if ent.description not in merged:
                    merged[ent.description] = norm_score

    sorted_tags = [{"description": k, "score": v} for k, v in sorted(merged.items(), key=lambda x: x[1], reverse=True)]
    logos = [logo.description for logo in response.logo_annotations]
    return sorted_tags, logos

# 画像から出たタグのリスト受け取り
def translate_missing_tags(tags):
    # 共有ボックス(state)から、翻訳キャッシュ辞書を取り出す
    trans_cache = state["trans_cache"]
    # 日本語の文字が含まれてないかつまだ翻訳キャッシュ辞書に登録されてない英単語を抽出
    needed = [
        t["description"] for t in tags
        if not any("\u3000" <= ch <= "\u9fff" for ch in t["description"])
        and t["description"] not in trans_cache
    ]
    # 未翻訳の英単語があれば一括翻訳
    if needed:
        # 重複を消す
        needed = list(set(needed))
        # Google Cloud Translation APIのクライアント作成
        translate_client = translate.Client()
        # 未翻訳リストを日本語に翻訳
        result = translate_client.translate(needed, target_language="ja")
        # 英語と翻訳結果をメモリ上の翻訳キャッシュ辞書に追加
        for en, res in zip(needed, result):
            trans_cache[en] = res["translatedText"]
        # サーバー再起動時も使えるようにtranslation_cache.jsonに保存
        save_json(TRANS_CACHE_JSON, trans_cache)
# 共有ボックス(state)からモデル、江東区DBを取り出し、英語・日本語の両ベクトルで類似度を計算、平均化。スコアが高い上位3件の品目・分別区分・備考・類似度を返す
def search_koto_rules_hybrid(query_en, query_ja, top_k=3):
    model = state["model"]
    koto_items = state["koto_items"]
    koto_embeddings = state["koto_embeddings"]

    vec_en = model.encode([query_en], normalize_embeddings=True)
    vec_ja = model.encode([query_ja], normalize_embeddings=True)
    sim_en = np.dot(koto_embeddings, vec_en.T).squeeze()
    sim_ja = np.dot(koto_embeddings, vec_ja.T).squeeze()
    blended_sim = (sim_en + sim_ja) / 2.0

    top_indices = np.argsort(blended_sim)[::-1][:top_k]
    return [{
        "item": koto_items[idx]["品目"],
        "category": koto_items[idx]["分別区分"],
        "memo": koto_items[idx]["備考"],
        "similarity": round(float(blended_sim[idx]), 4)
    } for idx in top_indices]
# 江東区DBの備考欄に「キャップ」など言葉が含まれていれば、Flutterの画面に表示する警告アラート文を自動生成する
def extract_parts_alerts(memo):
    if not memo or not isinstance(memo, str):
        return "-"
    alerts = []
    if "キャップ" in memo or "ラベル" in memo: alerts.append("⚠️ キャップとラベルを外す")
    if "汚れ" in memo or "すす" in memo or "洗っ" in memo: alerts.append("⚠️ 汚れを落とす（すすぐ）")
    if "電池" in memo: alerts.append("⚠️ 電池を抜く")
    if "中身" in memo or "空に" in memo: alerts.append("⚠️ 中身を使い切る")
    if "30cm" in memo: alerts.append("⚠️ 30cm以上は粗大ごみ")
    return " / ".join(alerts) if alerts else "特になし"

#　Flutterから画像を受け取るエンドポイント
@app.post("/classify")
async def classify_garbage(file: UploadFile = File(...)):
    # 送られてきた画像の中身を読み込む
    content = await file.read()
    # 読み込んだ画像をVision APIに投げ、タグ、ロゴ一覧を受け取る
    labels, logos = detect_labels_vision_bytes(content)
    brand_str = " ".join(logos) if logos else "-"

    # タグが1つも取れなければ特定不能と返す
    if not labels:
        return {"status": "特定不能（足切り）", "category": "不明", "item": "不明", "material": "不明", "brand": brand_str, "alert": "-", "memo": "-", "alternatives": [], "score": 0.0}

    # 未知のタグは翻訳、必要なものは共有ボックス(state)から取り出す
    translate_missing_tags(labels)
    model = state["model"]
    ng_embeddings = state["ng_embeddings"]
    exclude_words = state["exclude_words"]
    trans_cache = state["trans_cache"]
    # 本命タグと素材タグのリスト
    specific_tags, material_tags = [], []
    # Vision APIのタグ1つずつ確認し、自信度0.55未満、除外ワード、NG概念を弾き、素材ならmaterial_tags、生き残ったタグならspecific_tagsに振り分ける
    for lb in labels:
        desc_lower = lb["description"].lower().strip()
        if lb["score"] < 0.55 or desc_lower in exclude_words:
            continue
        if desc_lower in MATERIAL_LABELS:
            material_tags.append(lb)
            continue
        tag_vec = model.encode([desc_lower], normalize_embeddings=True)
        if float(np.max(np.dot(ng_embeddings, tag_vec.T))) >= 0.58:
            continue
        specific_tags.append(lb)

    # 素材タグを日本語に変換、重複を消してリスト化し、"/"で繋いだ文字列にする
    detected_materials = list(dict.fromkeys(
        MATERIAL_LABELS[m["description"].lower().strip()] for m in material_tags
    ))
    mat_str = " / ".join(detected_materials) if detected_materials else "不明"

    # タグの上位6つについての英語、日本語、スコアのペアを作る / 上位3つをスペースで繋げたパターン(スコアは平均を取る)も追加し、検索する
    best_candidate = None
    if specific_tags:
        top_n = specific_tags[:6]
        trials = [(t["description"], trans_cache.get(t["description"], t["description"]), t["score"]) for t in top_n]
        if len(top_n[:3]) > 1:
            comb_en = " ".join(t["description"] for t in top_n[:3])
            comb_ja = " ".join(trans_cache.get(t["description"], t["description"]) for t in top_n[:3])
            trials.append((comb_en, comb_ja, round(float(np.mean([t["score"] for t in top_n[:3]])), 4)))
        # 上位タグ6つを1つずつ検索にかけ、Visionスコア x 1位の類似度の総合スコアが高かった結果を保存する
        for q_en, q_ja, v_sc in trials:
            m_list = search_koto_rules_hybrid(q_en, q_ja, top_k=3)
            comb_sc = round(v_sc * m_list[0]["similarity"], 4)
            if best_candidate is None or comb_sc > best_candidate["combined"]:
                best_candidate = {"v_score": v_sc, "matches": m_list, "sim": m_list[0]["similarity"], "combined": comb_sc}
    # 品目タグより素材タグの自信度が0.15高ければ素材優先フラグにするが、類似度0.88以上かつ総合スコア0.50以上ならば素材優先フラグを強制的にオフにする
    prefer_material = bool(material_tags and best_candidate and (material_tags[0]["score"] - best_candidate["v_score"] >= 0.15))
    special_match = bool(best_candidate and best_candidate["sim"] >= 0.88 and best_candidate["combined"] >= 0.50)
    if special_match:
        prefer_material = False

    # Flutterに結果を返す
    # 条件クリアもしくは特例なら、1位の結果と2〜3位のリストをJSONに返す
    if best_candidate and not prefer_material and ((best_candidate["sim"] >= 0.75 and best_candidate["combined"] >= 0.65) or special_match):
        best = best_candidate["matches"][0]
        return {
            "status": "確定（マッチ）", "category": best["category"], "item": best["item"],
            "material": mat_str, "brand": brand_str, "alert": extract_parts_alerts(best["memo"]),
            "memo": best["memo"], "alternatives": best_candidate["matches"][1:], "score": best_candidate["combined"]
        }
    # 素材タグがある場合、素材名と参考候補をJSONに返す
    if material_tags:
        return {
            "status": "素材から選択", "category": "各素材の分別を確認", "item": detected_materials[0],
            "material": mat_str, "brand": brand_str, "alert": "素材別の分解ルールを確認",
            "memo": "-", "alternatives": best_candidate["matches"] if best_candidate else [], "score": 0.0
        }
    # 確定にもならず、素材タグもなかったが、江東区DBとの類似度が0.55以上ある場合、1位〜3位を候補としてJSONに返す
    if best_candidate and best_candidate["sim"] >= 0.55:
        best = best_candidate["matches"][0]
        return {
            "status": "候補から選択", "category": best["category"], "item": best["item"],
            "material": mat_str, "brand": brand_str, "alert": extract_parts_alerts(best["memo"]),
            "memo": best["memo"], "alternatives": best_candidate["matches"][1:], "score": best_candidate["combined"]
        }
    # 全て該当しない場合は特定不能とJSONに返す
    return {"status": "特定不能（足切り）", "category": "不明", "item": "不明", "material": mat_str, "brand": brand_str, "alert": "-", "memo": "-", "alternatives": [], "score": 0.0}
